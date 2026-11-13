// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! vibe-cluster — LLM cluster library.
//! Manages multiple LLM providers and distributes prompts across them.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use vibe_cluster::{
//!     CancelToken, Cluster, Connection, OllamaParams, Prompt, PromptEvent,
//! };
//!
//! let connections = vec![Connection::ollama(
//!     "http://localhost:11434",
//!     "qwen2.5-coder:3b-instruct",
//!     OllamaParams {
//!         temperature: Some(0.7),
//!         num_ctx: Some(32768),
//!         num_predict: Some(2048),
//!         ..Default::default()
//!     },
//! )];
//!
//! let cluster = Cluster::new(connections).unwrap();
//! let result = cluster
//!     .call(Prompt::new("Explain async/await", None, None))
//!     .await
//!     .unwrap();
//! println!("{}", result.text.unwrap());
//! ```
//!
//! # Streaming
//!
//! ```rust,ignore
//! use vibe_cluster::{
//!     CancelToken, Cluster, Connection, Params, OllamaParams, Prompt,
//!     PromptEvent, Provider,
//! };
//!
//! let connections = vec![Connection {
//!     provider: Provider::Ollama,
//!     host: "http://localhost:11434".into(),
//!     model: "qwen2.5-coder:3b-instruct".into(),
//!     params: Params::Ollama(OllamaParams {
//!         temperature: Some(0.7),
//!         num_ctx: Some(32768),
//!         num_predict: Some(2048),
//!         ..Default::default()
//!     }),
//!     ..Default::default()
//! }];
//! let cluster = Cluster::new(connections).unwrap();
//!
//! // Thinking and text deltas arrive as they are generated;
//! // the aggregated result comes back, cancellation cuts the stream.
//! let cancel = CancelToken::new();
//! let result = cluster
//!     .call_stream(Prompt::new("Explain async/await", None, None), &cancel, |event| {
//!         match event {
//!             PromptEvent::Thinking(chunk) => print!("{}", chunk),
//!             PromptEvent::Text(chunk) => print!("{}", chunk),
//!             _ => {}
//!         }
//!     })
//!     .await
//!     .unwrap();
//! let stats = result.stats.expect("a completed call carries the report");
//! println!("\n{} ms, {} tokens out", stats.duration_ms, stats.output_tokens.unwrap_or(0));
//! ```
//!
//! # Layout
//!
//! - `capability` — name-based capability detection: rules + defaults.
//! - `client` — the `ClientTrait` contract for provider APIs.
//! - `clients` — one implementation per provider API + the factory.
//! - `cluster` — the engine: distribution, retries, progress.
//! - `constants` — cross-client mechanics constants: timeouts, backoffs, protocol values.
//! - `data` — types exchanged between the cluster and its clients.
//! - `error` — the failure contract.
//! - `support` — internal client helpers (auth, http, stream); nothing here is public.
//! - `utils` — public helpers (image payloads, normalize, tokens).
//! - `validate` — the validation contract and per-type validators.

mod capability;
mod client;
mod clients;
mod cluster;
mod constants;
mod data;
mod error;
mod support;
mod utils;
mod validate;

pub use capability::CapabilityRules;
pub use client::ClientTrait;
pub use clients::GigaChatClient;
pub use clients::OllamaClient;
pub use clients::OpenAiClient;
pub use cluster::Cluster;
pub use data::Abilities;
pub use data::BatchOptions;
pub use data::Bpe;
pub use data::CancelToken;
pub use data::Capabilities;
pub use data::Connection;
pub use data::Effort;
pub use data::GigaChatParams;
pub use data::HistoryMessage;
pub use data::Input;
pub use data::ModelFilter;
pub use data::ModelInfo;
pub use data::OllamaParams;
pub use data::OpenAiParams;
pub use data::Output;
pub use data::Params;
pub use data::Prompt;
pub use data::PromptEvent;
pub use data::PromptResult;
pub use data::PromptStats;
pub use data::Provider;
pub use data::Role;
pub use error::ClusterError;
pub use validate::ValidateTrait;

pub use clients::create_client;
pub use utils::count_tokens;
pub use utils::mime_from_base64;
pub use utils::normalize_text;
pub use utils::strip_data_uri;
