// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! LLM provider types supported by vibe-cluster.
//! Each variant maps to a specific API implementation.

/// Provider type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Provider {
    Ollama,
    /// Any OpenAI-compatible host (api.openai.com by default): vLLM, OpenRouter, custom gateways.
    OpenAI,
    DeepSeek,
    Qwen,
    Moonshot,
    Zhipu,
    Yandex,
    GigaChat,
}

impl Provider {
    /// All supported providers, catalog order.
    pub const ALL: [Provider; 8] = [
        Provider::Ollama,
        Provider::OpenAI,
        Provider::DeepSeek,
        Provider::Qwen,
        Provider::Moonshot,
        Provider::Zhipu,
        Provider::Yandex,
        Provider::GigaChat,
    ];
}
