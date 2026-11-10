// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Token counting for prompts and answers.
//!
//! Counts are estimates: the provider's tokenizer differs from ours,
//! exact only for the matching encoding family. Good enough for usage
//! stats, not for billing.

use crate::data::Bpe;

/// Counts the tokens of the text with the selected BPE — exact for
/// the matching model family, an estimate for the rest.
pub fn count_tokens(text: &str, bpe: Bpe) -> usize {
    bpe.count(text)
}
