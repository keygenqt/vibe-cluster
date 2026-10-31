// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Heuristics for detecting vision (image input) support of models.
//!
//! The model name is lowercased and matched against a keyword list of the
//! specific provider. Keywords are verified against the official provider
//! documentation as of October 2026.
//!
//! Keywords are matched as whole "words": a match counts only when the key
//! is bounded by the start/end of the string or a separator
//! (`-`, `_`, `.`, `:`, `/`) — see [`has_key`]. For example, `vl` matches
//! `qwen3-vl-plus` but not inside `vllm`; `glm` matches `glm-4v` but not
//! `glm4`.

use crate::data::Provider;

// ============================================================================
// PUBLIC API
// ============================================================================

/// Returns `true` if the `model` of the `provider` accepts image input
/// (vision / multimodal).
///
/// The name case is ignored. Per-provider rules live in the
/// `*_supports_vision` functions below.
pub(crate) fn model_supports_vision(provider: &Provider, model: &str) -> bool {
    let name = model.to_ascii_lowercase();

    match provider {
        Provider::DeepSeek => deepseek_supports_vision(&name),
        Provider::OpenAI => openai_supports_vision(&name),
        Provider::Qwen => qwen_supports_vision(&name),
        Provider::Moonshot => moonshot_supports_vision(&name),
        Provider::Zhipu => zhipu_supports_vision(&name),
        Provider::Yandex => yandex_supports_vision(&name),
        Provider::GigaChat => gigachat_supports_vision(&name),
        Provider::Ollama => ollama_supports_vision(&name),
    }
}

// ============================================================================
// PROVIDERS
// ============================================================================

/// DeepSeek.
///
/// Only the Flash line accepts images: `deepseek-flash` has vision in the
/// API (JPEG/PNG/GIF/WebP), while the legacy names `deepseek-v4-flash` and
/// `deepseek-v4-flash-vision-exp` are served by the same Flash model.
/// `deepseek-v4-pro` and old text-only aliases (`deepseek-chat`, ...) have
/// no vision.
fn deepseek_supports_vision(name: &str) -> bool {
    // `flash`  — deepseek-flash, deepseek-v4-flash(-vision-exp)
    // `vision` — legacy names like *-vision-*
    const KEYS: &[&str] = &["flash", "vision"];

    has_any_key(name, KEYS)
}

/// Qwen (DashScope).
///
/// Vision lines: `qwen-vl-*`, `qwen-omni-*` (text + images + audio) and
/// `qvq-*` — the visual reasoning model.
fn qwen_supports_vision(name: &str) -> bool {
    // `vl`    — qwen-vl-max, qwen-vl-plus, qwen2.5-vl-*, ...
    // `omni`  — qwen-omni-turbo, ...
    // `qvq`   — qvq-max, qvq-plus, ...
    const KEYS: &[&str] = &["omni", "qvq", "vl"];

    has_any_key(name, KEYS)
}

/// Moonshot (Kimi).
///
/// Every current model of the platform (`kimi-k3`, `kimi-k3.1-pro`,
/// `kimi-k2.6`, `kimi-k2.7-code*`, `kimi-latest`) accepts images and
/// video. From the old `moonshot-v1-*` line only the variants with
/// `vision` in the name do (the rest are text-only and already retired
/// from the API).
fn moonshot_supports_vision(name: &str) -> bool {
    // `kimi`   — all current kimi-* (kimi-k3, kimi-k2.6, kimi-latest, ...)
    // `vision` — legacy: moonshot-v1-*-vision-preview
    const KEYS: &[&str] = &["kimi", "vision"];

    has_any_key(name, KEYS)
}

/// Zhipu (BigModel).
///
/// Vision models: the GLM-4V family (`glm-4v`, `glm-4v-plus`,
/// `glm-4v-flash`) and GLM-4.5V/4.6V — all of them have a `v` right after
/// a digit in the version number; plus the new natively multimodal
/// GLM-5.3-Flash / GLM-5.3-FlashX line. Text-only `glm-5.3`, `glm-5.2`,
/// `glm-4.5`, etc. do not pass the check.
fn zhipu_supports_vision(name: &str) -> bool {
    // `glm` — the whole GLM family, the version is refined below.
    const KEYS: &[&str] = &["glm"];
    // GLM-5.3-Flash / GLM-5.3-FlashX — multimodal without a "v" in the version.
    const FLASH_VL_KEYS: &[&str] = &["5.3-flash", "5.3-flashx"];

    has_any_key(name, KEYS) && (digit_v(name) || has_any_key(name, FLASH_VL_KEYS))
}

/// Yandex (Yandex Cloud AI Studio).
///
/// Native models (`yandexgpt-*`, `aliceai-llm*`) are text-only, and
/// Alice AI ART only generates images. Image input is supported by the
/// hosted third-party models: Qwen3.6 35B (works with Base64-encoded
/// images) and the DeepSeek Flash line — the latter is checked by the
/// DeepSeek rules.
fn yandex_supports_vision(name: &str) -> bool {
    // `qwen3.6-35b` — qwen3.6-35b-a3b works with Base64-encoded images
    const KEYS: &[&str] = &["qwen3.6-35b"];

    has_any_key(name, KEYS) || deepseek_supports_vision(name)
}

/// GigaChat (Sber).
///
/// Image input is accepted by the legacy dedicated models
/// (`GigaChat-Vision`, `GigaChat-Vision-Pro`), the GigaChat 2 Pro / Max
/// tiers and GigaChat 3 Ultra. `GigaChat-2-Lite` and the first-gen base
/// models are text-only.
fn gigachat_supports_vision(name: &str) -> bool {
    const KEYS: &[&str] = &["vision", "2-pro", "2-max", "3-ultra"];

    has_any_key(name, KEYS)
}

/// Ollama (local models).
///
/// Covers popular vision models of the library: LLaVA and its forks,
/// Gemma 3, MiniCPM-V, Moondream, Llama 3.2 Vision, Granite Vision,
/// Qwen-VL and the multimodal GLM-5.3-Flash.
fn ollama_supports_vision(name: &str) -> bool {
    // `bakllava`      — BakLLaVA (Mistral 7B + LLaVA)
    // `gemma3`        — Gemma 3 (4b/12b/27b multimodal; 1b is text-only)
    // `glm-5.3-flash` — multimodal GLM-5.3-Flash
    // `llava`         — llava, llava-llama3, llava-phi3, ...
    // `minicpm-v`     — MiniCPM-V
    // `moondream`     — Moondream 2
    // `omni`          — lfm3-omni, qwen*-omni
    // `vision`        — llama3.2-vision, granite3.2-vision, ...
    // `vl`            — qwen2-vl, lfm3-vl, ...
    const KEYS: &[&str] = &[
        "bakllava",
        "gemma3",
        "glm-5.3-flash",
        "llava",
        "minicpm-v",
        "moondream",
        "omni",
        "vision",
        "vl",
    ];
    // GLM: vision versions are marked with a digit + `v`
    // (`glm-4v`, `glm-4.5v`), while `glm4` and `glm-4.5` are text-only.
    const DIGIT_V_KEYS: &[&str] = &["glm"];

    has_any_key(name, KEYS) || (has_any_key(name, DIGIT_V_KEYS) && digit_v(name))
}

/// OpenAI.
///
/// Image input is supported by: the GPT-4o family (including
/// `gpt-4o-mini` and `chatgpt-4o-latest`), GPT-4.1 (including
/// `-mini`/`-nano`), GPT-4 Turbo with Vision, GPT-5 with all its variants,
/// and the o-models except the lightweight `o1-mini` / `o3-mini`, which do
/// not accept images.
///
/// DeepSeek, Qwen, Moonshot, Zhipu and Yandex go through the same
/// OpenAI-compatible client — their checks are reused here. GigaChat
/// (OAuth client) and Ollama (own API) are not included.
fn openai_supports_vision(name: &str) -> bool {
    const KEYS: &[&str] = &[
        "chatgpt-4o",
        "gpt-4-turbo",
        "gpt-4.1",
        "gpt-4o",
        "gpt-5",
        "o4",
    ];
    const EXACT: &[&str] = &["o1", "o1-preview", "o1-pro", "o3", "o3-pro"];

    has_any_key(name, KEYS)
        || EXACT.contains(&name)
        || deepseek_supports_vision(name)
        || qwen_supports_vision(name)
        || moonshot_supports_vision(name)
        || zhipu_supports_vision(name)
        || yandex_supports_vision(name)
}

// ============================================================================
// HELPERS
// ============================================================================

/// `true` if `name` contains at least one of the `keys`.
fn has_any_key(name: &str, keys: &[&str]) -> bool {
    keys.iter().any(|key| has_key(name, key))
}

/// Matches a key as a whole "word": the match counts only when the key is
/// bounded on both sides by the start/end of the string or a separator
/// (`-`, `_`, `.`, `:`, `/`).
///
/// So `vl` is found in `qwen3-vl-plus` but not inside `vllm`,
/// and `glm` is found in `glm-4v` but not in `glm4`.
fn has_key(name: &str, key: &str) -> bool {
    name.match_indices(key).any(|(start, _)| {
        let end = start + key.len();
        let left_boundary = start == 0 || is_separator(name.as_bytes()[start - 1]);
        let right_boundary = end == name.len() || is_separator(name.as_bytes()[end]);

        left_boundary && right_boundary
    })
}

/// Checks a model-name separator byte.
fn is_separator(byte: u8) -> bool {
    matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
}

/// `true` if the name has a digit immediately followed by `v` —
/// the vision-version marker (`glm-4v`, `glm-4.5v`).
fn digit_v(name: &str) -> bool {
    name.as_bytes()
        .windows(2)
        .any(|w| w[0].is_ascii_digit() && w[1] == b'v')
}
