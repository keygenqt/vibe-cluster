// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Crate data types exchanged between the cluster and its clients.
//!
//! # Submodules
//!
//! - `bpe` — optional BPE tokenizer for estimate-backed token counts.
//! - `cancel` — cancellation handle for in-flight requests.
//! - `capabilities` — model capabilities: modalities and abilities.
//! - `model_info` — a model as a provider host reports it.
//! - `prompt_event` — streamed pieces of a model response.
//! - `options` — batch execution options and the model filter.
//! - `params` — connection and generation settings for a single model.
//! - `prompt` — the prompt and the conversation history.
//! - `provider` — the provider kind.
//! - `result` — the result of a single prompt execution.
//! - `stats` — the token and timing report of a prompt execution.

mod bpe;
mod cancel;
mod capabilities;
mod model_info;
mod options;
mod params;
mod prompt;
mod prompt_event;
mod provider;
mod result;
mod stats;

pub use bpe::Bpe;
pub use cancel::CancelToken;
pub use capabilities::Abilities;
pub use capabilities::Capabilities;
pub use capabilities::Input;
pub use capabilities::Output;
pub use model_info::ModelInfo;
pub use options::BatchOptions;
pub use options::ModelFilter;
pub use params::Connection;
pub use params::GigaChatParams;
pub use params::OllamaParams;
pub use params::OpenAiParams;
pub use params::Params;
pub use prompt::HistoryMessage;
pub use prompt::Prompt;
pub use prompt::Role;
pub use prompt_event::PromptEvent;
pub use provider::Provider;
pub use result::PromptResult;
pub use stats::PromptStats;
