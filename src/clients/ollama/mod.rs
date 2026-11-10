// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Ollama native API client.
//!
//! # Submodules
//!
//! - `client` — the `OllamaClient` entity and its `Client` impl.
//! - `requests` — request bodies and the non-stream call paths.
//! - `capabilities` — live detection: `/api/show` → `Capabilities`.
//! - `stream` — streaming call paths and the NDJSON chunk forwarder.

mod call;
mod capabilities;
mod client;
mod stream;

pub use client::OllamaClient;
