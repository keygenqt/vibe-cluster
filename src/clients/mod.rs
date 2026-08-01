//! Internal HTTP clients for each supported API type.
//! Ollama uses its own API; OpenAI-compatible covers DeepSeek, Qwen, Kimi, Zhipu.

use crate::{
    clients::{client::Client, ollama::OllamaClient, openai::OpenAiClient},
    params::ConnectionParams,
    provider::Provider,
};

pub mod client;
mod ollama;
mod openai;
mod vision;

pub fn create_client(params: ConnectionParams) -> Box<dyn Client> {
    match params.provider {
        Provider::Ollama => Box::new(OllamaClient::new(params)),
        Provider::DeepSeek | Provider::Qwen | Provider::Kimi | Provider::Zhipu => {
            Box::new(OpenAiClient::new(params))
        }
    }
}
