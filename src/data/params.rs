// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Connection and generation parameters for LLM providers.
//! Used by Cluster to create internal clients for each model.

use crate::data::Bpe;
use crate::{constants, data::Provider};

/// Generation parameters of the Ollama API (the request `options`
/// block). `None` = not sent, the server/Modelfile default applies.
#[derive(Debug, Clone, Default)]
pub struct OllamaParams {
    /// Sampling temperature (0.0-2.0). Lower = more deterministic.
    pub temperature: Option<f64>,
    /// Seed for reproducible outputs.
    pub seed: Option<u64>,
    /// Nucleus sampling cutoff, probability mass (0.0-1.0).
    pub top_p: Option<f64>,
    /// Top-k sampling cutoff, kept most likely tokens.
    pub top_k: Option<u32>,
    /// Min-p sampling cutoff, scaled from the top token.
    pub min_p: Option<f64>,
    /// Stop sequences; generation ends when one is produced.
    pub stop: Option<Vec<String>>,
    /// Context window size in tokens.
    pub num_ctx: Option<usize>,
    /// Maximum tokens to generate.
    pub num_predict: Option<usize>,
}

/// Generation parameters of the OpenAI-compatible schema: OpenAI,
/// DeepSeek, Qwen, Moonshot, Zhipu, Yandex, local gateways (vLLM,
/// llama.cpp). `top_k`/`min_p` are outside the strict OpenAI schema —
/// local OpenAI-compatible hosts accept them.
/// `None` = not sent, the provider default applies.
#[derive(Debug, Clone, Default)]
pub struct OpenAiParams {
    /// Sampling temperature (0.0-2.0). Lower = more deterministic.
    pub temperature: Option<f64>,
    /// Seed for reproducible outputs.
    pub seed: Option<u64>,
    /// Nucleus sampling cutoff, probability mass (0.0-1.0).
    pub top_p: Option<f64>,
    /// Top-k sampling cutoff; not accepted by the strict OpenAI host.
    pub top_k: Option<u32>,
    /// Min-p sampling cutoff; not accepted by the strict OpenAI host.
    pub min_p: Option<f64>,
    /// Stop sequences; generation ends when one is produced.
    pub stop: Option<Vec<String>>,
    /// Maximum tokens to generate (`max_tokens` on the wire).
    pub num_predict: Option<usize>,
}

/// Generation parameters of the GigaChat schema. `seed` is not in the
/// documented parameter list but tolerated by the server.
/// `None` = not sent, the model default applies.
#[derive(Debug, Clone, Default)]
pub struct GigaChatParams {
    /// Sampling temperature (0.0-2.0). Lower = more deterministic.
    pub temperature: Option<f64>,
    /// Seed for reproducible outputs (undocumented, tolerated).
    pub seed: Option<u64>,
    /// Nucleus sampling cutoff, probability mass (0.0-1.0).
    pub top_p: Option<f64>,
    /// Maximum tokens to generate (`max_tokens` on the wire).
    pub num_predict: Option<usize>,
}

/// The generation parameters of a connection, per provider family.
/// The variant must match `Connection::provider` — a mismatch
/// is a config error. Wire names are mapped by the clients:
/// `num_predict` goes out as `max_tokens` on the OpenAI-compatible
/// and GigaChat schemas.
#[derive(Debug, Clone)]
pub enum Params {
    Ollama(OllamaParams),
    OpenAi(OpenAiParams),
    GigaChat(GigaChatParams),
}

/// The default family follows the default provider — Ollama.
impl Default for Params {
    fn default() -> Self {
        Self::Ollama(OllamaParams::default())
    }
}

/// Connection settings for a single model.
#[derive(Debug, Clone)]
pub struct Connection {
    /// Provider type: Ollama, DeepSeek, Qwen, etc.
    pub provider: Provider,
    /// API endpoint URL (e.g. http://localhost:11434 for Ollama, https://api.deepseek.com/v1 for DeepSeek).
    pub host: String,
    /// Model name (e.g. "qwen2.5:7b", "deepseek-chat").
    pub model: String,
    /// Generation parameters of the provider family.
    pub params: Params,
    /// Request timeout in seconds.
    pub timeout_secs: Option<u64>,
    /// API key for cloud providers (not needed for Ollama).
    pub api_key: Option<String>,
    /// Project/folder id for providers that route by it (Yandex).
    pub project: Option<String>,
    /// Number of parallel connections to this provider (default 1).
    /// Keep 1 for `BatchOptions::exclusive` — siblings break the
    /// eject guarantee.
    pub parallel: usize,
    /// Optional BPE estimator: fills the token counts the provider
    /// did not report. `None` — provider facts only.
    pub bpe: Option<Bpe>,
}

impl Connection {
    /// Ollama connection: provider is fixed, mechanics stay `Default`.
    pub fn ollama(host: impl Into<String>, model: impl Into<String>, params: OllamaParams) -> Self {
        Self {
            provider: Provider::Ollama,
            host: host.into(),
            model: model.into(),
            params: Params::Ollama(params),
            ..Default::default()
        }
    }

    /// OpenAI-compatible connection: OpenAI itself, DeepSeek, Qwen,
    /// Moonshot, Zhipu, Yandex, local gateways.
    pub fn open_ai(
        provider: Provider,
        host: impl Into<String>,
        model: impl Into<String>,
        params: OpenAiParams,
    ) -> Self {
        Self {
            provider,
            host: host.into(),
            model: model.into(),
            params: Params::OpenAi(params),
            ..Default::default()
        }
    }

    /// GigaChat connection.
    pub fn gigachat(model: impl Into<String>, params: GigaChatParams) -> Self {
        Self {
            provider: Provider::GigaChat,
            host: String::new(),
            model: model.into(),
            params: Params::GigaChat(params),
            ..Default::default()
        }
    }
}

/// Default connection: Ollama on localhost with 120s timeout, no model set.
impl Default for Connection {
    fn default() -> Self {
        Self {
            provider: Provider::Ollama,
            host: "http://localhost:11434".into(),
            model: String::new(),
            params: Params::default(),
            timeout_secs: Some(constants::DEFAULT_CALL_TIMEOUT_SECS),
            api_key: None,
            project: None,
            parallel: 1,
            bpe: None,
        }
    }
}
