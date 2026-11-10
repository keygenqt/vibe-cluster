// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Internal HTTP clients for each supported API type.
//! Ollama uses its own API; OpenAI-compatible covers OpenAI itself,
//! DeepSeek, Qwen, Moonshot, Zhipu and Yandex; GigaChat uses OAuth + OpenAI-compatible.
//!
//! # Submodules
//!
//! - `factory` — the provider factory: client creation and the params family check.
//! - `gigachat` — the GigaChat client.
//! - `ollama` — the Ollama client.
//! - `openai` — the OpenAI-compatible client.

mod factory;
mod gigachat;
mod ollama;
mod openai;

pub use factory::create_client;
pub use factory::create_client_with_rules;
pub use gigachat::GigaChatClient;
pub use ollama::OllamaClient;
pub use openai::OpenAiClient;
