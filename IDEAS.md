# IDEAS — vibe-cluster

A thinking backlog, not a roadmap: items are picked up when a real
need appears, dropped when it does not. Each item names the gap it
closes in the current design.

## Routing and execution

- Per-prompt failover across models. A prompt that dies with its
  client after retries is never re-queued to the remaining candidates —
  the one promise of the crate ("distributes across a pool") has this
  hole. Candidate shape: `BatchOptions::fallback: bool`.
- Context-window routing. `ModelInfo.context_length` is collected but
  never used in selection; a pre-call estimate via `count_tokens`
  could drop models the prompt cannot fit.
- Generalize vision routing to `required: Capabilities` in
  `BatchOptions` — model selection by any subset of the matrix
  (tools, thinking, audio) instead of the hardcoded image check in
  `prepare_clients`.
- Per-provider concurrency limits: a semaphore per host/provider so a
  10-worker batch does not hammer a cloud API with a 5 RPS limit.
- Graceful shutdown / drain: finish in-flight, take no new prompts.
  `CancelToken` is a hard cut today; a drain flag would suit
  long-running workers.

## Protocol and providers

- Structured output: native JSON mode (`format: json` for Ollama,
  `response_format` for OpenAI-compatible). `utils::normalize_text` is
  a workaround built on top of its absence — the native mode is the
  real fix.
- 429 and Retry-After: `RETRYABLE_STATUS_RANGE` covers 5xx only, cloud
  providers throttle with 429; the header value is ignored. A separate
  branch in `execute`, not just one more status in the range.
- Embeddings call. Detection exists (`abilities.embedding`, GigaChat
  parses `type: embedder`) but no client can call an embedder (Ollama
  `/api/embed`, OpenAI-compatible `/v1/embeddings`).
- Audio/video input: the matrix detects the modalities,
  `support::http::content_value` carries images only.
- GigaChat `/files` upload: the documented image path; inline base64 in
  `image_url` is unverified — the TODO is recorded in
  `clients/gigachat/call.rs`.
- Tool calling: `abilities.tools` is detected, the API cannot send
  `tools` or parse `tool_calls`. Big job — only if projects go into
  agents.

## Public API surface

- `serde::Serialize`/`Deserialize` on `Prompt`, `PromptResult`,
  `PromptStats` — pipelines that persist results (the `Prompt.key`
  hints they exist) cannot today.
- Cluster config from a file (JSON/TOML) or a builder — connection
  lists are hand-assembled vecs.
- Cluster-level streaming over a channel: `Cluster::call_stream` takes
  a callback; the client level returns a receiver. A channel variant
  would suit consumers driving their own loops.
- Prompt priority in the batch queue — the worker pool pulls indexes
  first-come.
- Sticky routing: pin a prompt key to the model that served it before.

## Observability and operations

- Retry/attempt events: retries surface only through `log::warn`; a
  consumer cannot hook "attempt 2 of 5 started".
- Session usage aggregation: totals of tokens and wall time per
  provider/model across a batch (buildable on `PromptResult`, cheap).
- Availability cache: `prepare_clients` probes every client per batch;
  a short-lived cache or background probe would cut the latency tail.
- `tracing` support behind a feature flag — `log` is the current,
  deliberate minimum.

## Quality and infrastructure

- Fuzz / property tests for the string-heavy paths: `normalize_text`
  and the SSE/NDJSON event splitters are regex and byte-slicing logic
  — the natural fuzz targets of the crate.
- Response caching / dedup: identical prompts in one batch hit the
  provider twice. Likely out of scope — a caller-side concern, listed
  to stay visible.
- Cost estimation with per-provider price tables: depends on prices
  rotting in the repo; probably a downstream concern.
- History window management: conversation history grows without a
  token-budget truncation helper; `count_tokens` is the ready base.
