# 数据导出、Token 统计与长度分桶说明

本文档从原始数据已经落盘开始，整理本仓库中使用的统一格式导出、抽样 token 估算和全量长度分桶流程。

## 1. 脚本与目录

| 路径 | 用途 |
| --- | --- |
| `dump_cpt_data.py` | 将原始记录统一导出为一行一个 `{"text": ...}` 的 JSONL，并可同时生成 FinePDFs Bernoulli 行采样。 |
| `data_processing/estimate_finepdf_tokens.py` | 从预生成的行采样中再抽取 shard，按字节比例估算 FinePDFs 总 token，并检查原目录名与实际 token 长度是否匹配。 |
| `data_processing/bucket_finepdfs_by_token_length.py` | 使用指定 `AutoTokenizer` 全量计算每条样本长度并写入 8 个长度桶；既支持原样搬运 JSONL，也支持从带元信息的 `.txt` JSONL 仅导出 `text`。脚本名保留历史命名。 |
| `outputs_stats/` | 最终 JSON、CSV、Markdown 统计，以及被 `.gitignore` 忽略的逐 shard 断点状态。 |

工作目录：

```bash
cd /afs-private/workspaces/wangweiyun/cos
```

主要 Python 依赖：

```text
transformers
tokenizers
orjson
```

## 2. 统一 Tokenizer 配置

本次统计和分桶统一使用：

```text
/afs/workspaces/wangweiyun/ckpts/gpt-oss-20b-BF16-vocab-extend-v2-general-sft-v4
```

配置如下：

| 配置 | 值 |
| --- | --- |
| 加载方式 | `AutoTokenizer.from_pretrained(path, local_files_only=True)` |
| tokenizer class | `TokenizersBackend` |
| vocab size | `200026` |
| `tokenizer.json` SHA-256 | `ba98a2c3a39652ce64fcef931ff8aa2e01fa0d734347a75664d0efcd193dd83f` |
| special token | `add_special_tokens=False` |
| 截断 | `truncation=False` |
| 输入字段 | JSON 对象的 `text` 字段 |

虽然 tokenizer 元数据中的 `model_max_length` 是 `262144`，worker 会把检查上限设为 `sys.maxsize`，并显式关闭截断。因此超过 256K、1M，甚至上亿 token 的样本都会完整编码。

核心计数方式：

```python
token_count = len(
    tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )
)
```

## 3. 统一导出为 `{"text": ...}` JSONL

`dump_cpt_data.py` 对不同源 schema 做以下转换：

- FinePDFs：要求 `len(messages) == 1`，导出 `messages[0]["content"]`。
- 普通 text 数据：导出原记录的 `text`。
- 输出格式：每行仅保留 `{"text": text}`。
- 跳过 `_SUCCESS` 和已存在正常文件对应的临时碎片。
- 每个目标文件先写临时文件，完成后用 `os.replace` 原子替换。

导出 FinePDFs，并同时生成 10% 确定性 Bernoulli 行采样：

```bash
python dump_cpt_data.py \
  --dataset finepdfs_20260703 \
  --jobs 8 \
  --sample-ratio 0.10 \
  --seed 20260703 \
  --progress-every 25
```

对应目录：

```text
输入：cpt-data/finepdfs_20260703
全量：cpt-data-dumped/finepdfs_20260703
采样：cpt-data-dumped/finepdfs_20260703_downsampled_10_percent
```

采样判断使用每个源文件独立且确定的随机序列：

```python
rng = random.Random(f"{seed}\0{src_path}")
selected = rng.random() < sample_ratio
```

因此相同 seed、文件路径和输入顺序会得到相同采样结果。

## 4. FinePDFs 抽样 Token 估算

该步骤用于在全量 token 化前快速估算总 token，并检查原 `block_section_*` 目录是否真的是 token 分桶。

实际执行命令：

```bash
python data_processing/estimate_finepdf_tokens.py \
  --data-root /afs-private/workspaces/wangweiyun/cos/cpt-data-dumped/finepdfs_20260703 \
  --sample-root /afs-private/workspaces/wangweiyun/cos/cpt-data-dumped/finepdfs_20260703_downsampled_10_percent \
  --tokenizer /afs/workspaces/wangweiyun/ckpts/gpt-oss-20b-BF16-vocab-extend-v2-general-sft-v4 \
  --files-per-bucket 10 \
  --workers 100 \
  --seed 20260714 \
  --output /afs-private/workspaces/wangweiyun/cos/outputs_stats/finepdfs_20260703_token_estimate.json
```

方法：

1. FinePDFs 的 10% 目录已经是逐行 Bernoulli 采样。
2. 在每个 `block_section_*` 中固定随机选择 10 个采样 shard。
3. 完整 token 化所选 shard 中的所有记录。
4. 每个原始目录单独按 JSONL 字节比例外推：

```text
estimated_bucket_tokens = sampled_tokens / sampled_jsonl_bytes * full_bucket_jsonl_bytes
```

5. 对 shard 级 token/byte 比例做 10,000 次 bootstrap，输出每个目录的稳定性区间。

本次抽样覆盖全量存储的 `0.998063%`，实测 `189049` 条、`1078361894` token，估算全量 `108079965332` token。后续全量精确结果为 `107997593440`，抽样误差约 `0.0763%`。

原始 FinePDFs 目录名最终确认表示 `block_id` 区间，不表示 token 长度。原始记录中的 `messages[0].token_count` 为 `-1`，所以必须重新 token 化后分桶。

## 5. 全量长度分桶

### 5.1 桶边界

`K` 和 `M` 使用二进制单位：`K=1024`，`M=1048576`。区间左闭右开，最后一桶无上界。

| 输出目录 | Token 区间 |
| --- | ---: |
| `0-32K` | `[0, 32768)` |
| `32K-64K` | `[32768, 65536)` |
| `64K-128K` | `[65536, 131072)` |
| `128K-256K` | `[131072, 262144)` |
| `256K-512K` | `[262144, 524288)` |
| `512K-768K` | `[524288, 786432)` |
| `768K-1M` | `[786432, 1048576)` |
| `1M-plus` | `[1048576, +inf)` |

### 5.2 输入布局

输入后缀由 `--input-suffix` 指定，默认是 `.jsonl`。脚本支持两种布局，但一次运行只选择其中一种：

```text
# shard 直接位于输入根目录
input_root/*<input-suffix>

# shard 位于输入根目录的一级子目录
input_root/*/*<input-suffix>
```

对于一级子目录布局，输出 shard 名会加上源子目录前缀，避免不同源目录下的同名文件冲突：

```text
block_section_0-100/part-00000-....jsonl
->
0-32K/block_section_0-100__part-00000-....jsonl
```

### 5.3 写出方式

- 一行 JSONL 是一条样本，必须包含字符串字段 `text`。
- 解析 JSON 只用于取得 `text` 并计算 token。
- 默认直接写原始 `line` 字节，不重新序列化，也不新增 `token_count` 字段。
- 指定 `--text-only-output` 时，重新序列化为仅含 `{"text": text}` 的 JSONL；可从原始数据单遍完成导出、精确 token 统计和分桶，不需要中间全量副本。
- 每个输入 shard 最多写出 8 个目标 shard。
- 所有目标 shard 先写隐藏临时文件，输入 shard 完成后再 `os.replace`。
- 默认原样写出模式要求单个 shard 的输入输出字节严格相等；`--text-only-output` 模式分别统计输入和输出字节。

核心逻辑等价于：

```python
item = orjson.loads(line)
token_count = len(tokenizer.encode(item["text"], add_special_tokens=False, truncation=False))
target_bucket = bucket_index(token_count)
writers[target_bucket].write(line)
```

仅导出 `text` 时，最后一行替换为：

```python
writers[target_bucket].write(orjson.dumps({"text": item["text"]}, option=orjson.OPT_APPEND_NEWLINE))
```

### 5.4 并发与内存控制

`--workers` 控制最大进程数。超长单条样本的 tokenizer 中间内存会远大于源 JSON 字节，因此可用 `--max-inflight-gib` 限制同时运行 shard 的源文件字节总和：

```text
--workers 128 --max-inflight-gib 16
```

调度器按文件从大到小处理。在 16 GiB 预算内，大文件阶段自动降低活动进程数；文件变小时逐步提高并发，最多使用 128 个进程。超长样本可能让 worker 的内存分配器保留大量 RSS，指定 `--recycle-workers` 后每个有界批次结束都会销毁进程池并释放内存。两个参数都只影响调度，不影响输出内容或断点配置摘要。

### 5.5 断点续跑

若没有显式传 `--state-root`，状态目录默认为：

```text
<stats-output 去掉 .json>_parts/
```

每个输入 shard 对应一个状态 JSON。状态中记录：

- 输入相对路径、字节数和 `mtime_ns`。
- 记录数、token 数、输出总字节。
- 每个长度桶的记录数、token 数、字节数、最小/最大长度和输出文件。
- tokenizer 路径及 SHA、桶边界、输入输出路径组成的配置摘要。

再次执行完全相同的命令时，脚本会验证状态和所有输出文件，只重算缺失或失效的 shard。配置摘要不一致时会直接报错，避免不同 tokenizer 或边界混写到同一个目录。

全量成功后会生成：

```text
<output-root>/_SUCCESS
<stats-output>.json
<stats-output>.csv
<stats-output>.md
```

## 6. 本次实际执行命令

### 6.1 FinePDFs

```bash
python data_processing/bucket_finepdfs_by_token_length.py \
  --input-root /afs-private/workspaces/wangweiyun/cos/cpt-data-dumped/finepdfs_20260703 \
  --output-root /afs-private/workspaces/wangweiyun/cos/cpt-data-dumped/finepdfs_20260703_by_token_length \
  --tokenizer /afs/workspaces/wangweiyun/ckpts/gpt-oss-20b-BF16-vocab-extend-v2-general-sft-v4 \
  --stats-output /afs-private/workspaces/wangweiyun/cos/outputs_stats/finepdfs_20260703_token_buckets.json \
  --workers 128 \
  --progress-every 5
```

结果：

```text
输入文件：1000
样本数：18909739
Token：107997593440
JSONL 字节：439268928458
运行时间：约 51 分钟
输出：cpt-data-dumped/finepdfs_20260703_by_token_length
统计：outputs_stats/finepdfs_20260703_token_buckets.{json,csv,md}
```

### 6.2 GitHub long_code_v1

```bash
python data_processing/bucket_finepdfs_by_token_length.py \
  --input-root /afs-private/workspaces/wangweiyun/cos/cpt-data-dumped/github-data-bj/260707/long_code_v1/all_repos_jsonl \
  --output-root /afs-private/workspaces/wangweiyun/cos/cpt-data-dumped/github-data-bj/260707/long_code_v1_by_token_length \
  --tokenizer /afs/workspaces/wangweiyun/ckpts/gpt-oss-20b-BF16-vocab-extend-v2-general-sft-v4 \
  --stats-output /afs-private/workspaces/wangweiyun/cos/outputs_stats/github_data_bj_260707_long_code_v1_token_buckets.json \
  --workers 128 \
  --progress-every 25
```

结果：

```text
输入文件：4780
样本数：24932
Token：8183319622
JSONL 字节：34704120661
最大单条：238874612 token
运行时间：约 17 分 51 秒
输出：cpt-data-dumped/github-data-bj/260707/long_code_v1_by_token_length
统计：outputs_stats/github_data_bj_260707_long_code_v1_token_buckets.{json,csv,md}
```

### 6.3 GitHub long_code_v3

源 `.txt` JSONL 包含 `instance_id`、`source`、`language_*`、`quality`、`meta` 等元信息。本次不先生成中间导出目录，而是单遍提取 `text`、精确 token 化并分桶：

```bash
python data_processing/bucket_finepdfs_by_token_length.py \
  --input-root /afs-private/workspaces/wangweiyun/cos/cpt-data/github-data-bj/260715/long_code_v3/all_repos_jsonl \
  --output-root /afs-private/workspaces/wangweiyun/cos/cpt-data-dumped/github-data-bj/260715/long_code_v3_by_token_length \
  --tokenizer /afs/workspaces/wangweiyun/ckpts/gpt-oss-20b-BF16-vocab-extend-v2-general-sft-v4 \
  --stats-output /afs-private/workspaces/wangweiyun/cos/outputs_stats/github_data_bj_260715_long_code_v3_token_buckets.json \
  --input-suffix .txt \
  --text-only-output \
  --workers 128 \
  --max-inflight-gib 16 \
  --recycle-workers \
  --progress-every 10
```

结果：

```text
输入文件：5348
样本数：209602
Token：95419234165
输入字节：383832580968
输出 JSONL 字节：376037506498
非空输出 shard：11576
最大单条：594648711 token
>=1M 桶：14080 条，74529167504 token（占总 token 78.1071%）
最终成功运行：约 1 小时 38 分（复用 56 份已完成状态）
输出：cpt-data-dumped/github-data-bj/260715/long_code_v3_by_token_length
统计：outputs_stats/github_data_bj_260715_long_code_v3_token_buckets.{json,csv,md}
```

## 7. 完整性检查

全量脚本内部执行以下检查：

1. 每个输入 shard 的读取字节数等于 `stat().st_size`。
2. 每个输入 shard 的输出字节等于 8 个桶的字节之和；原样写出模式还要求它等于输入字节数。
3. 聚合后的记录数和 token 数等于 8 个桶的求和。
4. 原样写出模式要求全部输出 JSONL 字节等于全部输入字节；`--text-only-output` 模式分别报告两者。
5. 只有全部检查通过后才写最终统计和 `_SUCCESS`。

本次还独立交叉验证了：

- 每个输入 shard 都有且只有一个断点状态文件。
- 状态中列出的每个输出文件都存在且字节数一致。
- 物理输出文件集合与状态记录集合完全一致。
- 每个桶的 `min_tokens`、`max_tokens` 都落在对应边界内。
- 不存在遗留的 `.*.tmp.*` 文件。

最终校验结果：

| 数据集 | 输入 shard | 非空输出 shard | 样本 | Token | 输入/输出字节 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FinePDFs | 1000 | 7397 | 18909739 | 107997593440 | 439268928458 / 439268928458 |
| long_code_v1 | 4780 | 16529 | 24932 | 8183319622 | 34704120661 / 34704120661 |
| long_code_v3 | 5348 | 11576 | 209602 | 95419234165 | 383832580968 / 376037506498 |

## 8. 运行注意事项

- 原样写出模式的输出体积与输入相同；`--text-only-output` 会去掉元信息，但仍应按接近完整输入体积预留空间。
- `--workers=128` 是在 256 CPU、约 4 TiB 内存机器上的实际配置；资源较小的机器应降低 worker 数。
- long-code v3 中存在约 2.10 GB 的单条 JSONL 记录。完整 tokenizer 编码时单 worker RSS 接近 300 GiB，必须配合 `--max-inflight-gib` 和 `--recycle-workers` 控制峰值并释放 allocator 保留内存。
- 不建议把超长文本随意切块后分别 token 化再相加，因为 BPE 在切分边界附近可能产生不同 token；本次结果全部来自完整样本编码。
- 删除 `<stats-output stem>_parts/` 不会删除最终数据，但会失去快速断点验证能力。
- 输入内容、mtime、tokenizer、桶边界或输出路径变化后，应使用新的输出和状态目录；不要混用旧状态。
- 空白 JSONL 行、无 `text` 字段或 `text` 非字符串会使任务立即失败，避免静默丢样本。

## 9. 结果文件索引

- `outputs_stats/finepdfs_20260703_token_estimate.json`
- `outputs_stats/finepdfs_20260703_token_buckets.json`
- `outputs_stats/finepdfs_20260703_token_buckets.csv`
- `outputs_stats/finepdfs_20260703_token_buckets.md`
- `outputs_stats/github_data_bj_260707_long_code_v1_token_buckets.json`
- `outputs_stats/github_data_bj_260707_long_code_v1_token_buckets.csv`
- `outputs_stats/github_data_bj_260707_long_code_v1_token_buckets.md`
- `outputs_stats/github_data_bj_260715_long_code_v3_token_buckets.json`
- `outputs_stats/github_data_bj_260715_long_code_v3_token_buckets.csv`
- `outputs_stats/github_data_bj_260715_long_code_v3_token_buckets.md`
