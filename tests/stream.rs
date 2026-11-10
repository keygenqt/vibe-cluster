// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Streaming tests: SSE/NDJSON chunk splitting on the mock server and
//! live cluster streaming with cancellation.

mod common;

use std::sync::Arc;

use vibe_cluster::CancelToken;
use vibe_cluster::Cluster;
use vibe_cluster::Connection;
use vibe_cluster::OpenAiParams;
use vibe_cluster::Params;
use vibe_cluster::Prompt;
use vibe_cluster::PromptEvent;
use vibe_cluster::Provider;
use vibe_cluster::create_client;

use crate::common::mock_server::mock_server;

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
    let client = create_client(Connection {
        provider: Provider::OpenAI,
        host,
        model: "gpt-4o".into(),
        params: Params::OpenAi(OpenAiParams::default()),
        ..Default::default()
    })
    .unwrap();

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
    let client = create_client(Connection {
        provider: Provider::Ollama,
        host,
        model: "test".into(),
        ..Default::default()
    })
    .unwrap();

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
    let cluster = Cluster::new(vec![Connection {
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

    // the final Ollama chunk reports the counters into the result
    let stats = result.stats.expect("a completed stream carries the report");
    assert!(stats.output_tokens.is_some(), "Ollama reports eval_count");

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
    let cluster = Cluster::new(vec![Connection {
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

#[tokio::test]
async fn test_sse_token_limit_with_no_text_is_stream_error() {
    let payload = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm\"}}]}\r\n\r\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\r\n\r\n",
        "data: [DONE]\r\n\r\n"
    );
    let (host, server) = mock_server(
        "200 OK",
        "text/event-stream",
        vec![payload.as_bytes().to_vec()],
    )
    .await;
    let client = create_client(Connection {
        provider: Provider::OpenAI,
        host,
        model: "gpt-4o".into(),
        params: Params::OpenAi(OpenAiParams {
            num_predict: Some(512),
            ..Default::default()
        }),
        ..Default::default()
    })
    .unwrap();

    let mut receiver = client
        .call_stream(&Prompt::new("test", None, None))
        .await
        .unwrap();
    let mut events = Vec::new();
    while let Some(event) = receiver.recv().await {
        events.push(event);
    }
    server.await.unwrap();

    // thinking went out; the empty limit cut is the same error the
    // non-stream paths report
    assert!(matches!(&events[0], Ok(PromptEvent::Thinking(chunk)) if chunk == "hmm"));
    assert!(matches!(
        events.last(),
        Some(Err(vibe_cluster::ClusterError::TokenLimitExceeded(message)))
            if message.contains("num_predict: 512")
    ));
}

#[tokio::test]
async fn test_ollama_token_limit_with_no_text_is_stream_error() {
    let payload = "{\"model\":\"test\",\"done\":true,\"done_reason\":\"length\"}\n";
    let (host, server) = mock_server(
        "200 OK",
        "application/x-ndjson",
        vec![payload.as_bytes().to_vec()],
    )
    .await;
    let client = create_client(Connection {
        provider: Provider::Ollama,
        host,
        model: "test".into(),
        ..Default::default()
    })
    .unwrap();

    let mut receiver = client
        .call_stream(&Prompt::new("test", None, None))
        .await
        .unwrap();
    let mut events = Vec::new();
    while let Some(event) = receiver.recv().await {
        events.push(event);
    }
    server.await.unwrap();

    assert!(matches!(
        events.last(),
        Some(Err(vibe_cluster::ClusterError::TokenLimitExceeded(_)))
    ));
}

#[tokio::test]
async fn test_sse_usage_chunk_reaches_stream() {
    let payload = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\r\n\r\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}}\r\n\r\n",
        "data: [DONE]\r\n\r\n"
    );
    let chunks = payload
        .as_bytes()
        .chunks(1)
        .map(|chunk| chunk.to_vec())
        .collect();
    let (host, server) = mock_server("200 OK", "text/event-stream", chunks).await;
    let client = create_client(Connection {
        provider: Provider::OpenAI,
        host,
        model: "gpt-4o".into(),
        params: Params::OpenAi(OpenAiParams::default()),
        ..Default::default()
    })
    .unwrap();

    let mut receiver = client
        .call_stream(&Prompt::new("test", None, None))
        .await
        .unwrap();
    let mut events = Vec::new();
    while let Some(event) = receiver.recv().await {
        events.push(event);
    }
    server.await.unwrap();

    assert!(matches!(&events[0], Ok(PromptEvent::Text(chunk)) if chunk == "ok"));
    assert!(matches!(
        events.last(),
        Some(Ok(PromptEvent::Usage {
            cached_tokens: None,
            input_tokens: Some(10),
            output_tokens: Some(2),
            thinking_tokens: None,
        }))
    ));
}

#[tokio::test]
async fn test_ollama_usage_counters_in_stream() {
    let payload = concat!(
        "{\"response\":\"ok\",\"done\":false}\n",
        "{\"response\":\"\",\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":21,\"prompt_eval_cached_count\":20,\"eval_count\":35}\n",
    );
    let chunks = payload
        .as_bytes()
        .chunks(1)
        .map(|chunk| chunk.to_vec())
        .collect();
    let (host, server) = mock_server("200 OK", "application/x-ndjson", chunks).await;
    let client = create_client(Connection {
        provider: Provider::Ollama,
        host,
        model: "test".into(),
        ..Default::default()
    })
    .unwrap();

    let mut receiver = client
        .call_stream(&Prompt::new("test", None, None))
        .await
        .unwrap();
    let mut events = Vec::new();
    while let Some(event) = receiver.recv().await {
        events.push(event);
    }
    server.await.unwrap();

    assert!(matches!(
        events.last(),
        Some(Ok(PromptEvent::Usage {
            cached_tokens: Some(20),
            input_tokens: Some(1),
            output_tokens: Some(35),
            thinking_tokens: None,
        }))
    ));
}

#[tokio::test]
async fn test_sse_partial_text_with_length_finish_is_not_an_error() {
    let payload = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\r\n\r\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\" answer\"}}]}\r\n\r\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\r\n\r\n",
        "data: [DONE]\r\n\r\n"
    );
    let chunks = payload
        .as_bytes()
        .chunks(1)
        .map(|chunk| chunk.to_vec())
        .collect();
    let (host, server) = mock_server("200 OK", "text/event-stream", chunks).await;
    let client = create_client(Connection {
        provider: Provider::OpenAI,
        host,
        model: "gpt-4o".into(),
        params: Params::OpenAi(OpenAiParams::default()),
        ..Default::default()
    })
    .unwrap();

    let mut receiver = client
        .call_stream(&Prompt::new("test", None, None))
        .await
        .unwrap();
    let mut text = String::new();
    while let Some(event) = receiver.recv().await {
        match event.unwrap() {
            PromptEvent::Text(chunk) => text.push_str(&chunk),
            _ => {}
        }
    }
    server.await.unwrap();

    // produced text means the limit cut is a truncated answer, not an error
    assert_eq!(text, "partial answer");
}
