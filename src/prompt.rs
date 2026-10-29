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
    /// Prior turns of the conversation, oldest first. Sent as native
    /// user/assistant messages before the current user prompt.
    pub history: Vec<HistoryMessage>,
}

/// One prior turn of the conversation.
#[derive(Debug, Clone)]
pub struct HistoryMessage {
    /// The author of the message: "user" or "assistant".
    pub role: String,
    /// The message text.
    pub content: String,
}

impl HistoryMessage {
    /// A message of the user.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    /// A message of the model.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
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
            history: Vec::new(),
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
            history: Vec::new(),
        }
    }
}
