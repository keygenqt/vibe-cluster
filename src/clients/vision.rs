//! Vision capability detection and image payload helpers.
//!
//! None of the supported cloud providers (DeepSeek, Qwen, Moonshot, Zhipu)
//! expose modality information through their OpenAI-compatible /models
//! endpoints, so vision support is detected by provider + model name.
//! Ollama performs live detection via /api/show (see OllamaClient) and
//! uses the name table here only as a last-resort fallback.
//!
//! NOTE: the name table must be updated when providers release new
//! vision-capable models that do not match the existing patterns.

use crate::provider::Provider;

/// Returns true if the given model of the given provider supports image input.
pub(crate) fn model_supports_vision(provider: &Provider, model: &str) -> bool {
    let name = model.to_lowercase();
    match provider {
        // DeepSeek cloud API currently has no vision-capable models
        // (deepseek-vl2 exists but is not served via api.deepseek.com).
        Provider::DeepSeek => false,

        // qwen-vl-*, qwen2-vl-*, qwen2.5-vl-*, qvq-* (vision-reasoning),
        // qwen-omni-* (multimodal)
        Provider::Qwen => name.contains("vl") || name.starts_with("qvq") || name.contains("omni"),

        // moonshot-v1-*-vision-preview, kimi-latest (multimodal)
        Provider::Moonshot => name.contains("vision") || name.starts_with("kimi-latest"),

        // glm-4v, glm-4v-plus, glm-4v-flash, glm-4.1v-thinking-flash, glm-4.5v
        Provider::Zhipu => digit_v(&name),

        // Yandex: text-only LLMs via the OpenAI endpoint; image generation is a separate API.
        Provider::Yandex => false,

        // GigaChat: no vision models on the personal scope.
        Provider::GigaChat => name.contains("vision"),

        // OllamaClient does live detection via /api/show; this arm is only
        // a name-based fallback (llava, qwen2.5-vl, minicpm-v, ...).
        Provider::Ollama => {
            name.contains("llava")
                || name.contains("vision")
                || name.contains("vl")
                || digit_v(&name)
        }
    }
}

/// True if the name contains a digit immediately followed by 'v':
/// matches glm-4v / glm-4.1v / glm-4.5v, but not deepseek-v3.
fn digit_v(name: &str) -> bool {
    name.as_bytes()
        .windows(2)
        .any(|w| w[0].is_ascii_digit() && w[1] == b'v')
}

/// Guesses the MIME type of a base64-encoded image by its prefix,
/// without decoding. Falls back to PNG.
pub(crate) fn mime_from_base64(img: &str) -> &'static str {
    if img.starts_with("/9j/") {
        "image/jpeg"
    } else if img.starts_with("iVBORw0KGgo") {
        "image/png"
    } else if img.starts_with("R0lGOD") {
        "image/gif"
    } else if img.starts_with("UklGR") {
        "image/webp"
    } else {
        "image/png"
    }
}

/// Strips a `data:<mime>;base64,` prefix if present.
/// Ollama wants raw base64; OpenAI-compatible APIs want a full data URI.
pub(crate) fn strip_data_uri(img: &str) -> &str {
    match img.find(',') {
        Some(i) if img[..i].starts_with("data:") => &img[i + 1..],
        _ => img,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zhipu_vision_models() {
        assert!(model_supports_vision(&Provider::Zhipu, "glm-4v"));
        assert!(model_supports_vision(&Provider::Zhipu, "glm-4v-plus"));
        assert!(model_supports_vision(&Provider::Zhipu, "glm-4v-flash"));
        assert!(model_supports_vision(
            &Provider::Zhipu,
            "glm-4.1v-thinking-flash"
        ));
        assert!(model_supports_vision(&Provider::Zhipu, "glm-4.5v"));
        assert!(!model_supports_vision(&Provider::Zhipu, "glm-4-plus"));
        assert!(!model_supports_vision(&Provider::Zhipu, "glm-4.5"));
    }

    #[test]
    fn qwen_vision_models() {
        assert!(model_supports_vision(&Provider::Qwen, "qwen-vl-max"));
        assert!(model_supports_vision(
            &Provider::Qwen,
            "qwen2.5-vl-72b-instruct"
        ));
        assert!(model_supports_vision(&Provider::Qwen, "qvq-max"));
        assert!(!model_supports_vision(&Provider::Qwen, "qwen-long"));
        assert!(!model_supports_vision(&Provider::Qwen, "qwen-max"));
    }

    #[test]
    fn moonshot_vision_models() {
        assert!(model_supports_vision(
            &Provider::Moonshot,
            "moonshot-v1-8k-vision-preview"
        ));
        assert!(model_supports_vision(&Provider::Moonshot, "kimi-latest"));
        assert!(!model_supports_vision(
            &Provider::Moonshot,
            "moonshot-v1-8k"
        ));
    }

    #[test]
    fn deepseek_has_no_vision() {
        assert!(!model_supports_vision(&Provider::DeepSeek, "deepseek-chat"));
        assert!(!model_supports_vision(
            &Provider::DeepSeek,
            "deepseek-reasoner"
        ));
        // digit_v must not false-positive on deepseek-v3 style names
        assert!(!model_supports_vision(&Provider::DeepSeek, "deepseek-v3"));
    }

    #[test]
    fn mime_detection() {
        assert_eq!(mime_from_base64("/9j/4AAQSkZJRg"), "image/jpeg");
        assert_eq!(mime_from_base64("iVBORw0KGgoAAAANS"), "image/png");
        assert_eq!(mime_from_base64("R0lGODlhAQAB"), "image/gif");
        assert_eq!(mime_from_base64("UklGRiQAAABXRUJQ"), "image/webp");
        assert_eq!(mime_from_base64("AAAA"), "image/png"); // fallback
    }

    #[test]
    fn data_uri_stripping() {
        assert_eq!(strip_data_uri("data:image/png;base64,iVBOR"), "iVBOR");
        assert_eq!(strip_data_uri("data:image/jpeg;base64,/9j/"), "/9j/");
        assert_eq!(strip_data_uri("iVBORw0KGgo"), "iVBORw0KGgo");
        assert_eq!(strip_data_uri("/9j/4AAQ"), "/9j/4AAQ");
    }
}
