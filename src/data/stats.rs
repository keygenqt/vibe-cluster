// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Usage stats of a prompt execution: what the client may extract
//! from the provider response and the resolved report of the call.

use crate::data::Prompt;
use crate::tokens::count_tokens;

/// Usage facts one client extracts from its provider response; the
/// fields the provider does not report stay `None` and fall back to
/// the BPE estimate. Today no client parses usage — the default
/// carries only `None`s.
#[derive(Debug, Clone, Default)]
pub struct ClientStats {
    /// Exact tokens of the user part (message and its images).
    pub user_tokens: Option<usize>,
    /// Exact tokens of the system prompt.
    pub system_tokens: Option<usize>,
    /// Exact tokens of the history together.
    pub history_tokens: Option<usize>,
    /// Exact tokens of the answer text.
    pub output_tokens: Option<usize>,
    /// Exact tokens of the reasoning output.
    pub thinking_tokens: Option<usize>,
    /// Exact tokens of the prompt served from the provider cache.
    /// The carrier for the future usage parsing; nothing reads it
    /// while no client reports usage.
    #[allow(dead_code)]
    pub cached_tokens: Option<usize>,
}

/// Token and timing report of one prompt execution. Every part is
/// the exact value the client reports, the BPE estimate of what was
/// fed otherwise. Everything fed to the request is counted: text and
/// image payloads alike — the image cost is its base64 size, not the
/// provider's image-token pricing. The estimates use the shared BPE:
/// exact for OpenAI, approximate for the rest.
#[derive(Debug, Clone, Default)]
pub struct PromptStats {
    /// Wall time of the whole call, ms: connection, retries,
    /// generation together.
    pub duration_ms: u64,
    /// Tokens of the current message with the inlined files and the
    /// turn's images.
    pub user_tokens: usize,
    /// Tokens of the system prompt.
    pub system_tokens: usize,
    /// Tokens of the conversation history together, images included.
    pub history_tokens: usize,
    /// Tokens of the answer text.
    pub output_tokens: usize,
    /// Tokens of the reasoning output.
    pub thinking_tokens: usize,
}

impl PromptStats {
    /// Measures one completed execution: every part is the exact
    /// value the client reports, the BPE estimate of what was fed
    /// otherwise. The caller measures `duration_ms` around the call.
    pub(crate) fn measure(
        duration_ms: u64,
        prompt: &Prompt,
        thinking: &str,
        text: &str,
        client: ClientStats,
    ) -> Self {
        let user = {
            let mut total = count_tokens(&prompt.user);
            for image in prompt.images.iter().flatten() {
                total += count_tokens(image);
            }
            total
        };
        let system = prompt.system.as_deref().map(count_tokens).unwrap_or(0);
        let history: usize = prompt
            .history
            .iter()
            .map(|message| {
                let mut total = count_tokens(&message.content);
                for image in &message.images {
                    total += count_tokens(image);
                }
                total
            })
            .sum();
        Self {
            duration_ms,
            user_tokens: client.user_tokens.unwrap_or(user),
            system_tokens: client.system_tokens.unwrap_or(system),
            history_tokens: client.history_tokens.unwrap_or(history),
            output_tokens: client.output_tokens.unwrap_or(count_tokens(text)),
            thinking_tokens: client.thinking_tokens.unwrap_or(count_tokens(thinking)),
        }
    }
}
