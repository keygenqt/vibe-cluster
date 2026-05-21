//! Prompt execution result returned by Cluster.
//! Contains response text, provider info, timing, and optional error.

use crate::{Prompt, error::ClusterError, provider::Provider};

/// Result of a single prompt execution.
#[derive(Debug, Clone)]
pub struct PromptResult {
    /// Provider that handled this prompt.
    pub prompt: Prompt,
    /// Provider that handled this prompt.
    pub provider: Provider,
    /// Host that served this prompt.
    pub host: String,
    /// Model name that generated the response.
    pub model: String,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// The response text (empty if error).
    pub text: Option<String>,
    /// Error if the prompt failed after all retries.
    pub error: Option<ClusterError>,
}
