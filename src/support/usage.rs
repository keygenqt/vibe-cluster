// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Usage parsing of the OpenAI-compatible `usage` node, shared by the
//! OpenAI-compatible clients (OpenAI, GigaChat, the MaaS aggregators).

use crate::data::PromptStats;

/// Parses the `usage` node of a chat completion response. Every field
/// is taken only from the node the provider actually sent:
///
/// - `cached_tokens`: `prompt_tokens_details.cached_tokens`, the
///   DeepSeek alias `prompt_cache_hit_tokens`, or the GigaChat alias
///   `precached_prompt_tokens` (no provider carries two of them);
/// - `input_tokens`: `prompt_tokens` minus the cached part;
/// - `output_tokens`: the Qwen detail `completion_tokens_details.text_tokens`
///   when present, else `completion_tokens` minus the reasoning tokens
///   (DeepSeek/Zhipu include them in the completion), else plain
///   `completion_tokens`;
/// - `thinking_tokens`: `completion_tokens_details.reasoning_tokens`.
pub(crate) fn parse_usage(usage: &serde_json::Value) -> PromptStats {
    let prompt = usage["prompt_tokens"].as_u64();
    let completion = usage["completion_tokens"].as_u64();

    let cached = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .or_else(|| usage["prompt_cache_hit_tokens"].as_u64())
        .or_else(|| usage["precached_prompt_tokens"].as_u64())
        .map(|value| value as usize);

    let input = match (prompt, cached) {
        (Some(prompt), Some(cached)) => Some(prompt.saturating_sub(cached as u64) as usize),
        (Some(prompt), None) => Some(prompt as usize),
        (None, _) => None,
    };

    let reasoning = usage["completion_tokens_details"]["reasoning_tokens"].as_u64();
    let output = usage["completion_tokens_details"]["text_tokens"]
        .as_u64()
        .or_else(|| match (completion, reasoning) {
            (Some(completion), Some(reasoning)) => Some(completion.saturating_sub(reasoning)),
            (Some(completion), None) => Some(completion),
            (None, _) => None,
        })
        .map(|value| value as usize);

    PromptStats::new(cached, input, output, reasoning.map(|value| value as usize))
}
