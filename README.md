# vibe-cluster

Multi-provider LLM cluster library for Rust.

## Why

Running a single local model hits limits fast. One GPU, one model, one prompt at a time. When you have multiple machines with different models, you're stuck managing connections, load balancing, and retries yourself.

Cloud APIs solve scaling but lock you into one provider. Local models give control but lack distribution.

One API, multiple providers, zero coordination overhead.

## What it does

- Distributes prompts across multiple LLM providers (local and cloud)
- Supports Ollama, OpenAI-compatible APIs (DeepSeek, Qwen, Moonshot, Zhipu, Yandex) and GigaChat (OAuth token + OpenAI-compatible)
- Vision model support — route image prompts to vision-capable models
- Parallel execution with automatic worker distribution
- Per-prompt results with provider info, timing, and error details
- Automatic retry on network errors and server failures (5xx)
- Streaming responses: thinking and text deltas as they arrive, via `call_stream`
- Cancellation: `CancelToken` cuts in-flight requests and the rest of a batch
- Model filtering: local only, cloud only, or specific models by name
- Exclusive mode: eject and reload model between prompts for clean context
- Smart worker selection: local models first, loaded models first
- Vision-aware routing — automatically filters clients when prompts contain images
- Batch call with progress reporting
- Response-to-prompt matching via optional keys

## Use cases

**Local LLM cluster.** Run models on multiple GPUs or machines. vibe-cluster distributes prompts and collects results. One API call instead of manual HTTP requests.

**Hybrid inference.** Combine local Ollama for fast drafts with cloud models for complex work. Filter by provider: `ModelFilter::LocalOnly` or `ModelFilter::CloudOnly`.

**Vision pipelines.** Send screenshots and images to vision models (minicpm-v, llava). Cluster automatically routes to vision-capable nodes.

**AI tooling.** Power code analysis pipelines, enrichment workers, and batch processing. Feed prompts in bulk, get structured results back.

**Testing and experimentation.** Run the same prompt across different models. Compare responses, measure latency, pick the best fit.

## Quick start

```bash
cargo add vibe-cluster
```

```rust
use vibe_cluster::{BatchOptions, Cluster, ConnectionParams, ModelFilter, Prompt, Provider};

// Configure connections
let connections = vec![
    ConnectionParams {
        provider: Provider::Ollama,
        host: "http://localhost:11434".into(),
        model: "qwen2.5-coder:3b-instruct".into(),
        ..Default::default()
    },
];

// Create cluster
let cluster = Cluster::new(connections).unwrap();

// Single prompt
let prompt = Prompt::new("Explain async/await in Rust", None, None);
let result = cluster.call(prompt).await.unwrap();
println!("{}", result.text.unwrap());

// Prompt with key for response matching
let prompt = Prompt::new_with_key("src/main.rs", "Analyze this code", None, None);

// Prompt with images for vision models
let prompt = Prompt {
    user: "Describe this image".into(),
    system: None,
    key: None,
    images: Some(vec!["iVBORw0KGgo...".into()]),
    history: Vec::new(),
};

// Single prompt with custom options
let options = BatchOptions {
    models: ModelFilter::LocalOnly,
    retries: Some(3),
    exclusive: true,
    ..Default::default()
};
let result = cluster
    .call_with_options(prompt, &options)
    .await
    .unwrap();

// Batch with progress
let prompts: Vec<Prompt> = files
    .iter()
    .map(|f| Prompt::new_with_key(f.path.clone(), build_prompt(f), None, None))
    .collect();
let results = cluster
    .batch_call(&prompts, |result, done, total| {
        println!("[{}/{}] {} in {}ms", done, total, result.model, result.duration_ms);
    })
    .await
    .unwrap();
```

| Provider | Type  | API                                                 | Vision (models)                       |
| -------- | ----- | --------------------------------------------------- | ------------------------------------- |
| Ollama   | Local | Native `/api/generate`                              | ✓ auto-detected via `/api/show`       |
| OpenAI   | Cloud | OpenAI-compatible `/v1` (api.openai.com by default) | ✓ `gpt-4o*`, `gpt-4.1*`, `o1`, `o3`   |
| DeepSeek | Cloud | OpenAI-compatible `/v1`                             | –                                     |
| Qwen     | Cloud | OpenAI-compatible `/v1`                             | ✓ `qwen-vl-*`, `qvq-*`, `qwen-omni-*` |
| Moonshot | Cloud | OpenAI-compatible `/v1`                             | ✓ `*-vision-preview`, `kimi-latest`   |
| Zhipu    | Cloud | OpenAI-compatible `/v1`                             | ✓ `glm-4v`, `glm-4.1v`, `glm-4.5v`    |
| Yandex   | Cloud | OpenAI-compatible `/v1`, `Api-Key` + project header | –                                     |
| GigaChat | Cloud | OAuth token + OpenAI-compatible `/v1`               | –                                     |

---

_Part of Vibe tools for command routing — [Vibe Action](https://vibe-action.keygenqt.com)._
