// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Optional BPE tokenizer for estimating the token counts the
//! provider did not report.

use std::sync::LazyLock;

use tiktoken_rs::CoreBPE;

/// BPE tokenizer of tiktoken-rs, one entry per model family: the
/// estimate is exact for the matching family, approximate for the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bpe {
    /// o200k_base — GPT-4o and newer OpenAI models.
    O200kBase,
    /// o200k_harmony — the gpt-oss models.
    O200kHarmony,
    /// cl100k_base — GPT-4 / GPT-3.5.
    Cl100kBase,
    /// p50k_base — text-davinci-003, Codex.
    P50kBase,
    /// r50k_base (gpt2) — legacy completions.
    R50kBase,
}

/// One BPE per process: building the ranks is expensive, counting is not.
static O200K_BASE: LazyLock<CoreBPE> =
    LazyLock::new(|| tiktoken_rs::o200k_base().expect("static o200k_base ranks are valid"));
static O200K_HARMONY: LazyLock<CoreBPE> =
    LazyLock::new(|| tiktoken_rs::o200k_harmony().expect("static o200k_harmony ranks are valid"));
static CL100K_BASE: LazyLock<CoreBPE> =
    LazyLock::new(|| tiktoken_rs::cl100k_base().expect("static cl100k_base ranks are valid"));
static P50K_BASE: LazyLock<CoreBPE> =
    LazyLock::new(|| tiktoken_rs::p50k_base().expect("static p50k_base ranks are valid"));
static R50K_BASE: LazyLock<CoreBPE> =
    LazyLock::new(|| tiktoken_rs::r50k_base().expect("static r50k_base ranks are valid"));

impl Bpe {
    /// Counts the tokens of the text with the selected encoding.
    pub fn count(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        let bpe = match self {
            Bpe::O200kBase => &*O200K_BASE,
            Bpe::O200kHarmony => &*O200K_HARMONY,
            Bpe::Cl100kBase => &*CL100K_BASE,
            Bpe::P50kBase => &*P50K_BASE,
            Bpe::R50kBase => &*R50K_BASE,
        };
        bpe.encode_with_special_tokens(text).len()
    }
}
