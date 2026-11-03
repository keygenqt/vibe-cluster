// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Token counting for prompts and answers.
//!
//! Counts are estimates: the o200k_base BPE is OpenAI's tokenizer, and
//! other providers tokenize differently. Good enough for usage stats,
//! not for billing.

use std::sync::LazyLock;
use tiktoken_rs::CoreBPE;

use crate::data::Prompt;

/// One BPE per process: building the ranks is expensive, counting is not.
static BPE: LazyLock<CoreBPE> =
    LazyLock::new(|| tiktoken_rs::o200k_base().expect("static o200k_base ranks are valid"));

/// Counts the tokens of the text with the o200k_base BPE — an
/// estimate for every provider, exact for OpenAI.
pub fn count_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    BPE.encode_with_special_tokens(text).len()
}

/// Counts the estimated tokens of the whole prompt: user, system
/// and history together.
pub fn count_prompt_tokens(prompt: &Prompt) -> usize {
    let mut total = count_tokens(&prompt.user);
    if let Some(system) = &prompt.system {
        total += count_tokens(system);
    }
    for message in &prompt.history {
        total += count_tokens(&message.content);
    }
    total
}
