// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Crate data types exchanged between the cluster and its clients.
//!
//! # Submodules
//!
//! - `chunk` — streamed pieces of a model response.
//! - `model_info` — a model as a provider host reports it.
//! - `options` — batch execution options and the model filter.
//! - `params` — connection settings for a single model.
//! - `prompt` — the prompt and the conversation history.
//! - `provider` — the provider kind.
//! - `result` — the result of a single prompt execution.

mod chunk;
mod model_info;
mod options;
mod params;
mod prompt;
mod provider;
mod result;

pub use chunk::PromptEvent;
pub use model_info::ModelInfo;
pub use options::BatchOptions;
pub use options::ModelFilter;
pub use params::ConnectionParams;
pub use prompt::HistoryMessage;
pub use prompt::Prompt;
pub use prompt::Role;
pub use provider::Provider;
pub use result::PromptResult;
