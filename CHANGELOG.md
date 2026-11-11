# Changelog

All notable changes to Vibe Cluster will be documented in this file.

## [0.3.0] - 2026-11-11

### 🐛 Fixes

- Exclude image payloads from token estimates

### 📚 Documentation

- Add IDEAS — a thinking backlog of possible crate extensions

### 🚀 Features

- Align capability rules with real provider model catalogs
- Per-call stats report with token breakdown by prompt part and wall time
- Add params families, provider usage token support, and the consistency pass

## [0.2.1] - 2026-11-07

### 🚀 Features

- Split create_client into embedded-rules default and create_client_with_rules

## [0.2.0] - 2026-11-07

### 🚀 Features

- Optional token counting (ConnectionParams.stats) into PromptResult
- Add image content support to chat history and vision-capable clients
- Capability rules engine with embedded JSON tables, Capabilities matrix, and connection validation

## [0.1.0] - 2026-10-31

### ⚡ Refactoring

- Improve error handling and checks

### 🐛 Fixes

- Up description

### 🚀 Features

- Add LLM cluster library with multi-provider support
- Add system prompt support to vibe-cluster
- Async models/hosts/providers methods, system prompt support for call()
- Remove error prefixes, strip markdown code blocks from responses, improve error messages
- Add vision model support and refactor clients
- Add kimi and zhipu providers with vision support
- Add model listing support to clients and rename kimi provider to moonshot
- Add conversation history support to ollama and openai clients and prompts
- Add Yandex and GigaChat providers, structured logging
- Full refactoring, add support yandex, gigachat, stream, etc

