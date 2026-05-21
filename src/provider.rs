//! LLM provider types supported by vibe-cluster.
//! Each variant maps to a specific API implementation.

/// Provider type.
#[derive(Debug, Clone, PartialEq)]
pub enum Provider {
    Ollama,
    DeepSeek,
    Qwen,
}
