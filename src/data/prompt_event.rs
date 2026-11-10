// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Streamed pieces of a model response.

/// One streamed piece of a model response: providers send thinking
/// and text as separate delta streams (Ollama NDJSON chunks,
/// OpenAI-compatible SSE `reasoning_content` / `content` deltas).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PromptEvent {
    /// A piece of the reasoning (thinking) output.
    Thinking(String),
    /// A piece of the response text.
    Text(String),
    /// Usage facts reported by the provider in the final chunk. The
    /// cluster intercepts this event into the final `PromptResult`
    /// and never passes it to the stream callback.
    Usage {
        /// Tokens the provider served from its cache.
        cached_tokens: Option<usize>,
        /// Uncached input tokens as the provider billed them.
        input_tokens: Option<usize>,
        /// Tokens of the answer text.
        output_tokens: Option<usize>,
        /// Tokens of the reasoning output.
        thinking_tokens: Option<usize>,
    },
}
