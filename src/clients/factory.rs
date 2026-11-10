// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! The provider factory: creates the client matching the provider kind.

use std::sync::Arc;

use crate::GigaChatClient;
use crate::OllamaClient;
use crate::OpenAiClient;
use crate::capability::CapabilityRules;
use crate::client::ClientTrait;
use crate::data::Connection;
use crate::data::Params;
use crate::data::Provider;
use crate::error::ClusterError;

/// Creates the client matching the provider kind with the embedded
/// capability rules.
pub fn create_client(params: Connection) -> Result<Box<dyn ClientTrait>, ClusterError> {
    create_client_with_rules(params, Arc::new(CapabilityRules::new()))
}

/// Creates the client matching the provider kind with the given
/// capability rules: they extend or replace the detection tables.
/// The params variant must match the provider family.
pub fn create_client_with_rules(
    connection: Connection,
    rules: Arc<CapabilityRules>,
) -> Result<Box<dyn ClientTrait>, ClusterError> {
    let params = connection.params.clone();
    match connection.provider {
        Provider::Ollama => match params {
            Params::Ollama(params) => Ok(Box::new(OllamaClient::new(connection, params, rules))),
            _ => Err(params_mismatch(&connection.model)),
        },
        Provider::OpenAI
        | Provider::DeepSeek
        | Provider::Qwen
        | Provider::Moonshot
        | Provider::Zhipu
        | Provider::Yandex => match params {
            Params::OpenAi(params) => Ok(Box::new(OpenAiClient::new(connection, params, rules))),
            _ => Err(params_mismatch(&connection.model)),
        },
        Provider::GigaChat => match params {
            Params::GigaChat(params) => {
                Ok(Box::new(GigaChatClient::new(connection, params, rules)))
            }
            _ => Err(params_mismatch(&connection.model)),
        },
    }
}

/// The params variant does not match the provider family.
fn params_mismatch(model: &str) -> ClusterError {
    ClusterError::Config(format!(
        "Connection '{}': params does not match the provider family.",
        model
    ))
}
