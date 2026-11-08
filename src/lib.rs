// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! vibe-cluster — LLM cluster library.
//! Manages multiple LLM providers and distributes prompts across them.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use vibe_cluster::{CancelToken, Cluster, ConnectionParams, Prompt, PromptEvent, Provider};
//!
//! let connections = vec![ConnectionParams {
//!     provider: Provider::Ollama,
//!     host: "http://localhost:11434".into(),
//!     model: "qwen2.5-coder:3b-instruct".into(),
//!     ..Default::default()
//! }];
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
//! use vibe_cluster::{CancelToken, Cluster, ConnectionParams, Prompt, PromptEvent, Provider};
//!
//! let connections = vec![ConnectionParams {
//!     provider: Provider::Ollama,
//!     host: "http://localhost:11434".into(),
//!     model: "qwen2.5-coder:3b-instruct".into(),
//!     stats: true,
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
//!         }
//!     })
//!     .await
//!     .unwrap();
//! let stats = result.stats.expect("stats: true fills the report");
//! println!("\n{} ms, {} tokens out", stats.duration_ms, stats.output_tokens);
//! ```
//!
//! # Layout
//!
//! - `capability` — name-based capability detection: rules + defaults.
//! - `cluster` — the engine: distribution, retries, progress.
//! - `client` — the `Client` trait contract for provider APIs.
//! - `clients` — one implementation per provider API + the factory.
//! - `data` — types exchanged between the cluster and its clients.
//! - `error` — the failure contract.
//! - `normalize` — response normalization helpers.
//! - `support` — shared client helpers (image payloads, SSE).
//! - `tokens` — token counting.
//! - `validate` — the validation contract and per-type validators.

mod cancel;
mod capability;
mod client;
mod clients;
mod cluster;
mod data;
mod error;
mod normalize;
mod support;
mod tokens;
mod validate;

pub use cancel::CancelToken;
pub use capability::CapabilityRules;
pub use client::Client;
pub use clients::GigaChatClient;
pub use clients::OllamaClient;
pub use clients::OpenAiClient;
pub use cluster::Cluster;
pub use data::Abilities;
pub use data::BatchOptions;
pub use data::Capabilities;
pub use data::ConnectionParams;
pub use data::HistoryMessage;
pub use data::Input;
pub use data::ModelFilter;
pub use data::ModelInfo;
pub use data::Output;
pub use data::Prompt;
pub use data::PromptEvent;
pub use data::PromptResult;
pub use data::PromptStats;
pub use data::Provider;
pub use data::Role;
pub use error::ClusterError;
pub use validate::ValidateTrait;

pub use clients::create_client;
pub use normalize::normalize_text;
pub use support::mime_from_base64;
pub use support::strip_data_uri;
pub use tokens::count_tokens;
