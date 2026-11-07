// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Internal HTTP clients for each supported API type.
//! Ollama uses its own API; OpenAI-compatible covers OpenAI itself,
//! DeepSeek, Qwen, Moonshot, Zhipu and Yandex; GigaChat uses OAuth + OpenAI-compatible.

use std::sync::Arc;

use crate::capability::CapabilityRules;
use crate::client::Client;
use crate::data::ConnectionParams;
use crate::data::Provider;

mod gigachat;
mod ollama;
mod openai_compatible;

pub use gigachat::GigaChatClient;
pub use ollama::OllamaClient;
pub use openai_compatible::OpenAiClient;

/// Creates the client matching the provider kind with the embedded
/// capability rules.
pub fn create_client(params: ConnectionParams) -> Box<dyn Client> {
    create_client_with_rules(params, Arc::new(CapabilityRules::new()))
}

/// Creates the client matching the provider kind with the given
/// capability rules: they extend or replace the detection tables.
pub fn create_client_with_rules(
    params: ConnectionParams,
    rules: Arc<CapabilityRules>,
) -> Box<dyn Client> {
    match params.provider {
        Provider::Ollama => Box::new(OllamaClient::new(params, rules.clone())),
        Provider::OpenAI
        | Provider::DeepSeek
        | Provider::Qwen
        | Provider::Moonshot
        | Provider::Zhipu
        | Provider::Yandex => Box::new(OpenAiClient::new(params, rules.clone())),
        Provider::GigaChat => Box::new(GigaChatClient::new(params, rules)),
    }
}
