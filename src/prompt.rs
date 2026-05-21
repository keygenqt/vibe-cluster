//! Prompt structure for cluster communication.
//! Contains the text to send and an optional key for response matching.

#[derive(Debug, Clone)]
pub struct Prompt {
    /// The prompt text sent to the model.
    pub text: String,
    /// Optional key for matching response back to source files.
    pub key: Option<String>,
}

impl Prompt {
    /// Create a new prompt with just text, no key.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            key: None,
        }
    }

    /// Create a new prompt with text and a matching key.
    pub fn new_with_key(text: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            key: Some(key.into()),
        }
    }
}
