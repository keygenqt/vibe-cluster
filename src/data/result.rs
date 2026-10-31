// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Prompt execution result returned by Cluster.
//! Contains response text, provider info, timing, and optional error.

use crate::data::{Prompt, Provider};
use crate::error::ClusterError;

/// Result of a single prompt execution.
#[derive(Debug)]
pub struct PromptResult {
    /// The prompt that produced this result.
    pub prompt: Prompt,
    /// Provider that handled this prompt.
    pub provider: Provider,
    /// Host that served this prompt.
    pub host: String,
    /// Model name that generated the response.
    pub model: String,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Optional chain of thought from reasoning models.
    pub thinking: Option<String>,
    /// The response text (empty if error).
    pub text: Option<String>,
    /// Error if the prompt failed after all retries.
    pub error: Option<ClusterError>,
}
