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
}

impl Provider {
    /// All supported providers, catalog order.
    pub const ALL: [Provider; 5] = [
        Provider::Ollama,
        Provider::DeepSeek,
        Provider::Qwen,
        Provider::Moonshot,
        Provider::Zhipu,
    ];
}
