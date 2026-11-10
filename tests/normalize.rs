// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! normalize_text tests.

use vibe_cluster::normalize_text;

#[test]
fn test_normalize_json_fence_and_backticks() {
    assert_eq!(normalize_text("```json\n{\"a\":1}\n```"), "{\"a\":1}");

    let raw = r#"{"text":"```"}"#;
    assert_eq!(normalize_text(raw), raw);
}

#[test]
fn test_normalize_single_line_unquoted_keys() {
    assert_eq!(normalize_text("{a:1}"), "{\"a\":1}");
    assert_eq!(normalize_text("{\"a\":1,b:2}"), "{\"a\":1,\"b\":2}");
    assert_eq!(normalize_text("{\"a\":1, b : 2}"), "{\"a\":1, \"b\": 2}");
}
