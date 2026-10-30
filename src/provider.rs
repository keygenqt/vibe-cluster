//! LLM provider types supported by vibe-cluster.
//! Each variant maps to a specific API implementation.

/// Provider type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Provider {
    Ollama,
    DeepSeek,
    Qwen,
    Moonshot,
    Zhipu,
    Yandex,
    GigaChat,
}

impl Provider {
    /// All supported providers, catalog order.
    pub const ALL: [Provider; 7] = [
        Provider::Ollama,
        Provider::DeepSeek,
        Provider::Qwen,
        Provider::Moonshot,
        Provider::Zhipu,
        Provider::Yandex,
        Provider::GigaChat,
    ];
}
