// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Image payload normalization helpers.
//! Prompt.images accepts raw base64 or a ready-made data: URI;
//! each client normalizes the payload for its provider.

/// Guesses the MIME type of a base64-encoded image by its prefix,
/// without decoding. Falls back to PNG.
pub fn mime_from_base64(img: &str) -> &'static str {
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
pub fn strip_data_uri(img: &str) -> &str {
    match img.find(',') {
        Some(i) if img[..i].starts_with("data:") => &img[i + 1..],
        _ => img,
    }
}
