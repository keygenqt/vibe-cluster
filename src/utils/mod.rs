// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Public helpers shipped to the crate users; everything here is
//! re-exported from the root `lib.rs`.
//!
//! # Submodules
//!
//! - `image` — image payload helpers: MIME detection, data URI stripping.
//! - `normalize` — response normalization.
//! - `tokens` — token counting.

mod image;
mod normalize;
mod tokens;

pub use image::mime_from_base64;
pub use image::strip_data_uri;
pub use normalize::normalize_text;
pub use tokens::count_tokens;
