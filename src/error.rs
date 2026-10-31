// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Error types for vibe-cluster operations.

/// Errors that can occur during model operations.
#[derive(Debug)]
pub enum ClusterError {
    /// Transport error before an HTTP response was received.
    Http(reqwest::Error),
    /// HTTP endpoint returned a non-success status.
    HttpStatus {
        /// HTTP status code.
        status: u16,
        /// Provider error message or response body.
        message: String,
    },
    /// Model returned an error response.
    Model(String),
    /// Failed to parse model response.
    Parse(String),
    /// Provider is not available.
    Unavailable(String),
    /// Invalid configuration.
    Config(String),
    /// All providers failed.
    AllFailed(String),
    /// Model does not support vision (images passed to non-vision model).
    VisionNotSupported(String),
    /// Model exhausted its output token limits before completing response.
    TokenLimitExceeded(String),
    /// The request was cancelled through the cancel token.
    Cancelled,
}

/// Human-readable error messages for cluster operations.
impl std::fmt::Display for ClusterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClusterError::Http(e) => write!(f, "{}", e),
            ClusterError::HttpStatus { status, message } => {
                write!(f, "HTTP {} - {}", status, message)
            }
            ClusterError::Model(e) => write!(f, "{}", e),
            ClusterError::Parse(e) => write!(f, "{}", e),
            ClusterError::Unavailable(e) => write!(f, "{}", e),
            ClusterError::Config(e) => write!(f, "{}", e),
            ClusterError::AllFailed(e) => write!(f, "{}", e),
            ClusterError::VisionNotSupported(e) => write!(f, "{}", e),
            ClusterError::TokenLimitExceeded(e) => write!(f, "{}", e),
            ClusterError::Cancelled => write!(f, "Request cancelled"),
        }
    }
}

/// Standard error trait — no wrapped source error.
impl std::error::Error for ClusterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

/// Converts reqwest HTTP errors into ClusterError::Http.
impl From<reqwest::Error> for ClusterError {
    fn from(e: reqwest::Error) -> Self {
        ClusterError::Http(e)
    }
}

/// Converts serde_json parse errors into ClusterError::Parse.
impl From<serde_json::Error> for ClusterError {
    fn from(e: serde_json::Error) -> Self {
        ClusterError::Parse(e.to_string())
    }
}

/// Formats the error with its source chain: reqwest reports only the
/// top-level message, the TLS/network cause sits deeper.
pub(crate) fn error_chain(error: &dyn std::error::Error) -> String {
    let mut chain = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        chain.push_str(" <- ");
        chain.push_str(&cause.to_string());
        source = cause.source();
    }
    chain
}
