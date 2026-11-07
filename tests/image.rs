// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Image payload normalization helper tests.

use vibe_cluster::mime_from_base64;
use vibe_cluster::strip_data_uri;

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
