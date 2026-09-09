# 04. Search batch / parallel search tracking

## 结论

当前训练配置使用的工具名是 `wiki_search`，不是任务描述中的泛化标签 `<search>`：

```text
<wiki_search>query_1 | query_2</wiki_search>
```

从代码可以可靠恢复“逻辑 search batch”的文本定义，但当前 live reward scorer 没有收到结构化 tool metadata，且“一个逻辑 batch”在活动 `SearchTool` 中会被拆成多个单 query HTTP 调用。因此需要区分三种计数：

| 计数对象 | 当前是否可得 | 含义 |
|---|---|---|
| logical search batch | 可从格式正确的 response/parser 输入中计算 | 一个完整的 `<wiki_search>...</wiki_search>` block |
| parsed query/tool-call count | 可从 parser 或 `detailed_tool_metrics` 计算 | 一个 block 按 `|` 拆出的非空 query 数之和 |
| actual HTTP request / attempt count | 当前 reward scorer 不可可靠得到 | 去重、错误和 HTTP retry 会使它与 parsed count 不同 |

这也是为什么本次审计对“能否统计 search batches”的最终状态标记为 `NEED_RUNTIME_VERIFICATION`：逻辑指标路径存在，但正式使用前仍需验证 response 序列化和实际 metadata 对齐。

## 1. Parse 链路

### 1.1 工具标签和 block 识别

活动工具配置是：

```text
verl/verl/tools/config/search_tool_config/wiki_rag_config.yaml:1-19
  class_name: verl.tools.search_tool.SearchTool
  function.name: wiki_search
```

XML parser 在 `verl/verl/tools/xml_tool_parser.py:30-44` 为配置中的每个工具建立正则。搜索工具集合为 `wiki_search` 和 `web_search`（27），正则使用 `re.DOTALL`，所以一个 block 可以跨多行。`parse_non_stream()` 位于 85-135：

1. 找到所有已配置工具标签；
2. 按文本位置排序；
3. 对 `wiki_search` / `web_search` 的内容调用 `_parse_search_queries()`；
4. 每个 query 生成一个 `ParsedToolCall`。

因此，当前代码中的“一个 search batch”最接近“一个被 parser 正则匹配到、且至少包含一个非空 query 的 search block”。它不是一个最终 DataProto 字段。

### 1.2 `|` 的处理

`_parse_search_queries()` 位于 `xml_tool_parser.py:63-79`：

```python
queries = [query.strip() for query in content.split('|')]
return [q for q in queries if q]
```

所以当前活动实现的定义是：

```text
num_search_batches      = 非空、格式正确的 wiki_search block 数
num_queries             = 所有 block 中非空 | 片段数之和
parallel_batch_count    = query 片段数 >= 2 的 block 数
parallel_sample         = parallel_batch_count > 0
parallel_factor         = num_queries / num_search_batches（batch 数 > 0）
```

`get_wiki_search_count()`（167-193）只返回 query 总数；没有现成的 batch count、parallel factor 或 query-range 字段。

## 2. 执行链路和“并行”语义

解析后，`sglang_rollout.py:812-853` 遍历 `parsed_tool_calls`，为每个 parsed call 构造一个 `tool.execute(...)` coroutine，然后在 836 使用 `asyncio.gather()`。这意味着一个 XML block 中的多个 query 会并发执行。

但在当前 GAP wiki 配置中，这不是一个 HTTP multi-query batch：

1. XML parser 将 `query_1 | query_2` 拆成两个 `ParsedToolCall`；
2. 每个 call 的参数是 `{"query": "..."}`（`xml_tool_parser.py:115-122`）；
3. 活动 `SearchTool.execute()`（`search_tool.py:201-235`）把单个 query 包装为 `[query]`；
4. `perform_single_search_batch()`（`search_r1_like_utils.py:130-214`）最终调用搜索 API，payload 中 `queries` 长度为 1。

因此：

```text
一个 XML parallel block
  = 一个逻辑 parallel batch
  = 多个并发的单 query SearchTool 调用
  ≠ 一个带多个 queries 的实际 HTTP 请求
```

`perform_single_search_batch()` 本身支持多 query list，但活动 XML -> `SearchTool` 路径没有把一个 block 作为一个 list 传进去。不要把这个函数名中的 “batch” 直接解释成当前 rollout 的 batch 计数。

## 3. observation 和 metadata 如何回到 trajectory

每个 tool call 的结果在 `sglang_rollout.py:839-853` 被加入 tool response message，并更新：

- `_req.messages`：通过 `schemas.py:183-191` 写入 `Message(role="tool", content=...)`；
- `_req.metrics`：`schemas.py:193-199` 按工具保存 metrics；
- `_req.tool_call_info`：按工具计数；
- `_req.detailed_tool_metrics`：`schemas.py:201-247` 保存总调用数、工具类型、assistant turns 和 `tool_call_sequence`。

rollout 输出在 `sglang_rollout.py:1142-1270` 组装为：

```text
batch:
  prompts, responses, input_ids, attention_mask, position_ids, loss_mask
non_tensor_batch:
  messages, reward_scores, complete_reason, finish_reason,
  tool_call_info, detailed_tool_metrics
```

`detailed_tool_metrics` 的 `tool_call_sequence` 包含 `tool_name`、`call_id`、`turn` 和 `arguments`。其中 parser 生成的 `call_id` 类似 `call_wiki_search_<block-index>_<query-index>`，在原则上能辅助恢复 block/query 关系；但当前 custom scorer 没有收到这个字段。`BatchRewardManager.verify()` 只向 scorer 转发 `questions`、`ground_truths`、`responses`、`data_sources`、`prompts` 和 `extra_infos`（`batch.py:34-68`）。

## 4. 当前已有字段能否直接回答目标指标

| 目标字段 | 当前是否存在同名字段 | 从 response text 计算 | 从现有结构化 metadata 计算 | 结论 |
|---|---:|---:|---:|---|
| `num_search_batches` | 否 | 格式正常时可以 | 当前没有直接字段；可由 `call_id` 约定或新增字段恢复 | 逻辑可行，需 runtime 验证 |
| `num_queries` | `tool_call_info` 可近似为 total parsed calls | 可以按 `|` 片段计数 | `detailed_tool_metrics.total_tool_calls` 可得 parsed count | 不是实际 HTTP 请求数 |
| `parallel_batch_count` | 否 | 可以 | 当前 metadata 未显式保存 block count | 可加轻量字段 |
| `parallel_sample` | 否 | 可以 | `call_id`/query 参数可能间接判断 | 可加轻量字段 |
| `parallel_factor` | 否 | 可以 | 先恢复前两项即可 | 可计算 |
| 成功 HTTP 请求数 | 否 | 不可以 | 需要每次 execute 的 status/duplicate/错误和 request id | 当前不可可靠恢复 |

## 5. Edge cases

### `|` 出现在普通 query 中

当前 parser 无转义规则，所有 `|` 都会被 `split('|')`。如果 query 的自然语言内容本身包含竖线，它会被误判为 parallel query。仅靠最终 response text 无法区分“模型想表达竖线”还是“模型发出了多个 query”。

### malformed XML / 未闭合 block

parser 的正则要求闭合标签。未闭合 `<wiki_search>` 不会产生 `ParsedToolCall`；仅用字符串计数则可能把它误算成一个 block。生产统计应同时记录 `parse_success`，而不是只对原文做 `count('<wiki_search>')`。

### 空 search block

`has_tool_call()` 只在内容非空时返回 true（`xml_tool_parser.py:46-55`），`_parse_search_queries()` 也会过滤空片段。因此 `<wiki_search></wiki_search>` 不会触发实际 tool call。报告中的 `num_search_batches` 应默认按“非空且成功 parse 的 block”定义，而不是按裸标签数量。

### 多行 query

正则使用 `DOTALL`，多行内容会被匹配；随后 `strip()` 保留中间换行。因此多行本身不是 parser 错误，但 query 的规范化和去重语义需要 runtime 检查。

### retry

`call_search_api()`（`search_r1_like_utils.py:34-116`）对连接、timeout 和部分 5xx 最多重试 10 次。`tool_call_info` / `detailed_tool_metrics` 记录的是 parsed tool call，不是 retry attempt。因此若 cost 定义为模型发出的 search action，重试不应加到 `num_search_batches`；若 cost 定义为网络请求成本，则现有字段不足。

### duplicate / search retry

`sglang_rollout.py:818-828` 使用 `web_search_history` 对相同 query 去重；重复 query 返回 canned response，不调用真实 search API，但 仍在 840-853 被计入 tool call 和 `tool_call_sequence`。因此 parsed query count 可能大于实际 HTTP request count。reflection 后再次搜索相同 query 也落入该情况。

### tool error

`SearchTool.execute()` 会把 API/执行错误放到返回 metrics（`search_tool.py:232-240`），调用仍会被记录到 tool-call 统计。当前 `detailed_tool_metrics` 没有按 call 保存 status/error；`_req.metrics` 虽然存在于 request，却没有进入 reward scorer。需要把 error/status 一并导出，才能定义“有效 search cost”。

### observation 中出现 `<wiki_search>` 字样

`BatchRewardManager` 解码的是完整 response token 序列，因此对最终文本做正则时无法天然区分真正的 XML tool call、模型引用的字符串和某些 observation 内容。rollout worker 在生成阶段也依赖正则 parser，而不是 XML AST。文本解析路径必须在 runtime 样本上和 `tool_call_sequence` 对齐。

## 6. 对新 reward 的建议

如果目标只是研究“模型产生了多少逻辑 search action”，可以先用 response text 做审计指标，并把结果标成 `logical_*`。如果目标是研究真实 retrieval 成本，建议在 rollout request 层新增每个 search block 的结构化记录，例如：

```text
search_batch_id
query_count
queries
parsed_call_count
executed_http_count
duplicate_count
status/error
retry_attempts
```

然后把 request-level summary 放进 `detailed_tool_metrics`，再通过 reward manager 转给 custom scorer。当前最小可行的 group-relative prototype 仍可以只解析 response，但正式实验前必须完成第 07 节的 runtime logging。

