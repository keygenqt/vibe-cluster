//! Error types for vibe-cluster operations.

/// Errors that can occur during model operations.
#[derive(Debug, Clone)]
pub enum ClusterError {
    /// HTTP transport error (connection, timeout).
    Http(String),
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
}

/// Human-readable error messages for cluster operations.
impl std::fmt::Display for ClusterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClusterError::Http(e) => write!(f, "HTTP error: {}", e),
            ClusterError::Model(e) => write!(f, "Model error: {}", e),
            ClusterError::Parse(e) => write!(f, "Parse error: {}", e),
            ClusterError::Unavailable(e) => write!(f, "Provider unavailable: {}", e),
            ClusterError::Config(e) => write!(f, "Config error: {}", e),
            ClusterError::AllFailed(e) => write!(f, "All providers failed: {}", e),
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
        ClusterError::Http(e.to_string())
    }
}

/// Converts serde_json parse errors into ClusterError::Parse.
impl From<serde_json::Error> for ClusterError {
    fn from(e: serde_json::Error) -> Self {
        ClusterError::Parse(e.to_string())
    }
}
