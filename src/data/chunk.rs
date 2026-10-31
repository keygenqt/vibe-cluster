// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Streamed pieces of a model response.

/// One streamed piece of a model response: providers send thinking
/// and text as separate delta streams (Ollama NDJSON chunks,
/// OpenAI-compatible SSE `reasoning_content` / `content` deltas).
#[derive(Debug, Clone)]
pub enum PromptEvent {
    /// A piece of the reasoning (thinking) output.
    Thinking(String),
    /// A piece of the response text.
    Text(String),
}
