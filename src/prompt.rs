//! Prompt structure for cluster communication.
//! Contains the text to send and an optional key for response matching.

#[derive(Debug, Clone)]
pub struct Prompt {
    /// User prompt text.
    pub user: String,
    /// Optional system prompt text.
    pub system: Option<String>,
    /// Optional key for matching response back to source files.
    pub key: Option<String>,
}

impl Prompt {
    /// Create prompt with user text and optional system text.
    pub fn new(user: impl Into<String>, system: Option<String>) -> Self {
        Self {
            user: user.into(),
            system,
            key: None,
        }
    }

    /// Create prompt with user text, key, and optional system text.
    pub fn new_with_key(
        user: impl Into<String>,
        key: impl Into<String>,
        system: Option<String>,
    ) -> Self {
        Self {
            user: user.into(),
            system,
            key: Some(key.into()),
        }
    }
}
