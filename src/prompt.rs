//! Prompt structure for cluster communication.
//! Contains the text to send and an optional key for response matching.

#[derive(Debug, Clone)]
pub struct Prompt {
    /// Optional key for matching response back to source files.
    pub key: Option<String>,
    /// User prompt text.
    pub user: String,
    /// Optional system prompt text.
    pub system: Option<String>,
    /// Base64-encoded images for vision models. None if not a vision request.
    pub images: Option<Vec<String>>,
}

impl Prompt {
    /// Create prompt with user text and optional system text.
    pub fn new(
        user: impl Into<String>,
        system: Option<String>,
        images: Option<Vec<String>>,
    ) -> Self {
        Self {
            key: None,
            user: user.into(),
            system,
            images,
        }
    }

    /// Create prompt with user text, key, and optional system text.
    pub fn new_with_key(
        key: impl Into<String>,
        user: impl Into<String>,
        system: Option<String>,
        images: Option<Vec<String>>,
    ) -> Self {
        Self {
            key: Some(key.into()),
            user: user.into(),
            system,
            images,
        }
    }
}
