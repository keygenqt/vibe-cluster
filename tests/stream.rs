// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Streaming tests: SSE/NDJSON chunk splitting on the mock server and
//! live cluster streaming with cancellation.

mod common;

use std::sync::Arc;

use common::mock_server;
use vibe_cluster::CancelToken;
use vibe_cluster::CapabilityRules;
use vibe_cluster::Cluster;
use vibe_cluster::ConnectionParams;
use vibe_cluster::Prompt;
use vibe_cluster::PromptEvent;
use vibe_cluster::Provider;
use vibe_cluster::create_client;

#[tokio::test]
async fn test_openai_stream_split_utf8_crlf_and_final_event() {
    let payload = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\r\n\r\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\" 👋\"}}]}"
    );
    let chunks = payload
        .as_bytes()
        .chunks(1)
        .map(|chunk| chunk.to_vec())
        .collect();
    let (host, server) = mock_server("200 OK", "text/event-stream", chunks).await;
    let client = create_client(
        ConnectionParams {
            provider: Provider::OpenAI,
            host,
            model: "gpt-4o".into(),
            ..Default::default()
        },
        Arc::new(CapabilityRules::new()),
    );

    let mut receiver = client
        .call_stream(&Prompt::new("test", None, None))
        .await
        .unwrap();
    let mut text = String::new();
    while let Some(event) = receiver.recv().await {
        if let PromptEvent::Text(chunk) = event.unwrap() {
            text.push_str(&chunk);
        }
    }
    server.await.unwrap();

    assert_eq!(text, "Hello 👋");
}

#[tokio::test]
async fn test_ollama_stream_split_utf8_and_final_line() {
    let payload = concat!(
        "{\"response\":\"Hello\",\"done\":false}\n",
        "{\"response\":\" 👋\",\"done\":true}"
    );
    let chunks = payload
        .as_bytes()
        .chunks(1)
        .map(|chunk| chunk.to_vec())
        .collect();
    let (host, server) = mock_server("200 OK", "application/x-ndjson", chunks).await;
    let client = create_client(
        ConnectionParams {
            provider: Provider::Ollama,
            host,
            model: "test".into(),
            ..Default::default()
        },
        Arc::new(CapabilityRules::new()),
    );

    let mut receiver = client
        .call_stream(&Prompt::new("test", None, None))
        .await
        .unwrap();
    let mut text = String::new();
    while let Some(event) = receiver.recv().await {
        if let PromptEvent::Text(chunk) = event.unwrap() {
            text.push_str(&chunk);
        }
    }
    server.await.unwrap();

    assert_eq!(text, "Hello 👋");
}

#[tokio::test]
async fn test_stream_call() {
    let cluster = Cluster::new(vec![ConnectionParams {
        provider: Provider::Ollama,
        host: "http://localhost:11434".into(),
        model: "qwen2.5-coder:3b-instruct".into(),
        timeout_secs: Some(60),
        ..Default::default()
    }])
    .unwrap();
    let cancel = CancelToken::new();

    let events: Arc<std::sync::Mutex<Vec<PromptEvent>>> = Arc::default();
    let events_seen = events.clone();

    let result = cluster
        .call_stream(
            Prompt::new("Reply with just: ok", None, None),
            &cancel,
            move |event| events_seen.lock().unwrap().push(event.clone()),
        )
        .await
        .unwrap();

    assert!(result.error.is_none());
    assert!(result.text.as_ref().unwrap().to_lowercase().contains("ok"));

    // the deltas the callback saw must assemble into the final text
    let mut assembled = String::new();
    for event in events.lock().unwrap().iter() {
        if let PromptEvent::Text(chunk) = event {
            assembled.push_str(chunk);
        }
    }
    assert_eq!(assembled, result.text.unwrap());
}

#[tokio::test]
async fn test_stream_cancel() {
    let cluster = Cluster::new(vec![ConnectionParams {
        provider: Provider::Ollama,
        host: "http://localhost:11434".into(),
        model: "qwen2.5-coder:3b-instruct".into(),
        timeout_secs: Some(60),
        ..Default::default()
    }])
    .unwrap();

    // cancelling before the call makes the stream cut immediately
    let cancel = CancelToken::new();
    cancel.cancel();

    let result = cluster
        .call_stream(
            Prompt::new("Reply with just: ok", None, None),
            &cancel,
            |_| {},
        )
        .await
        .unwrap();

    assert!(matches!(
        result.error,
        Some(vibe_cluster::ClusterError::Cancelled)
    ));
}
