// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Usage stats of a prompt execution: what the provider reported
//! about the call, optionally completed with BPE estimates.

use crate::data::Bpe;
use crate::data::Prompt;

/// Token and timing report of one prompt execution. Token fields are
/// the exact values the provider reported (`None` = not reported);
/// with `Connection.bpe` set, the missing ones are filled with BPE
/// estimates of the prompt and the response texts. Images never enter
/// the estimates — their real cost is the provider's image-token
/// pricing, and only the usage can tell it.
///
/// Known overlap: Ollama reports `eval_count` (output) including the
/// thinking tokens, while the thinking estimate counts them separately —
/// with estimates enabled the sum is inflated by the thinking size.
#[derive(Debug, Clone, Default)]
pub struct PromptStats {
    /// Wall time of the whole call, ms: connection, retries,
    /// generation together. Measured by the executor.
    pub duration_ms: u64,
    /// Tokens the provider served from its cache; the cheap part of
    /// the input. Never estimated — the cache cannot be counted
    /// from the outside.
    pub cached_tokens: Option<usize>,
    /// Uncached input tokens: system, history, the current message
    /// and its images together, as the provider billed them.
    pub input_tokens: Option<usize>,
    /// Tokens of the answer text.
    pub output_tokens: Option<usize>,
    /// Tokens of the reasoning output.
    pub thinking_tokens: Option<usize>,
}

impl PromptStats {
    /// A report of the facts one client extracted from the provider
    /// response; the executor fills `duration_ms` after the call.
    pub(crate) fn new(
        cached_tokens: Option<usize>,
        input_tokens: Option<usize>,
        output_tokens: Option<usize>,
        thinking_tokens: Option<usize>,
    ) -> Self {
        Self {
            duration_ms: 0,
            cached_tokens,
            input_tokens,
            output_tokens,
            thinking_tokens,
        }
    }

    /// Fills the fields the provider did not report with BPE estimates.
    /// `cached_tokens` stays untouched — the cache is not countable
    /// from the outside.
    pub(crate) fn estimate_missing(
        &mut self,
        bpe: Bpe,
        prompt: &Prompt,
        thinking: &str,
        text: &str,
    ) {
        if self.input_tokens.is_none() {
            let system = prompt.system.as_deref().map(|s| bpe.count(s)).unwrap_or(0);
            let history: usize = prompt.history.iter().map(|m| bpe.count(&m.content)).sum();
            self.input_tokens = Some(system + history + bpe.count(&prompt.user));
        }
        if self.output_tokens.is_none() {
            self.output_tokens = Some(bpe.count(text));
        }
        if self.thinking_tokens.is_none() {
            self.thinking_tokens = Some(bpe.count(thinking));
        }
    }
}
