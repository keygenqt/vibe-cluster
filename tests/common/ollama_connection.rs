// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Local Ollama connection presets for the live tests.

use vibe_cluster::Connection;
use vibe_cluster::OllamaParams;
use vibe_cluster::Params;
use vibe_cluster::Provider;

/// One local Ollama connection with the deterministic test params.
fn ollama_connection(model: &str) -> Connection {
    Connection {
        provider: Provider::Ollama,
        host: "http://localhost:11434".into(),
        model: model.into(),
        params: Params::Ollama(OllamaParams {
            temperature: Some(0.1),
            seed: Some(42),
            num_ctx: Some(4096),
            num_predict: Some(2048),
            ..Default::default()
        }),
        timeout_secs: Some(60),
        ..Default::default()
    }
}

pub fn test_config() -> Vec<Connection> {
    vec![ollama_connection("qwen2.5-coder:3b-instruct")]
}

pub fn test_config_two_models() -> Vec<Connection> {
    vec![
        ollama_connection("qwen2.5-coder:3b-instruct"),
        ollama_connection("gemma4:12b"),
    ]
}
