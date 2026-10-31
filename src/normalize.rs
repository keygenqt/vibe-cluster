// SPDX-FileCopyrightText: 2026 Vitaliy Zarubin <keygenqt@yandex.ru>
// SPDX-License-Identifier: Apache-2.0

//! Response normalization utilities.
//! Extract JSON from markdown fences, fix malformed JSON, return text as-is otherwise.

/// Normalize LLM response: strip edge markdown code blocks from text/JSON, fix malformed JSON.
pub fn normalize_text(raw: &str) -> String {
    let mut content = raw.trim().to_string();

    // Strip leading and trailing markdown code block markers (e.g., ```markdown ... ```)
    // from plain text responses, ensuring the trailing marker is removed only if the leading one was present.
    let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    let mut has_markdown_prefix = false;

    if let Some(first) = lines.first() {
        let trimmed = first.trim_start();
        if trimmed.starts_with("```markdown") || trimmed == "```md" {
            has_markdown_prefix = true;
        }
    }

    if has_markdown_prefix {
        lines.remove(0);
        if let Some(last) = lines.last() {
            if last.trim_start().starts_with("```") {
                lines.pop();
            }
        }
        content = lines.join("\n").trim().to_string();
    }

    if !content.contains('{') && !content.contains('[') {
        return content;
    }

    // Valid JSON may contain markdown markers inside strings.
    if serde_json::from_str::<serde_json::Value>(&content).is_ok() {
        return content;
    }

    if !content.starts_with('[')
        && !content.starts_with('{')
        && !content.starts_with("```[")
        && !content.starts_with("```{")
        && !content.starts_with("```json")
        && !content.starts_with("```\n[")
        && !content.starts_with("```\n{")
    {
        return content;
    }

    // Extract content from markdown code block (Only processed if it actually contains a JSON string inside)
    if let Some(start_md) = content.find("```") {
        let after_start = &content[start_md + 3..];
        let body_start = after_start.find('\n').map(|i| i + 1).unwrap_or(0);
        let body = &after_start[body_start..];
        if let Some(end_md) = body.find("```") {
            content = body[..end_md].trim().to_string();
        } else {
            content = body.trim().to_string();
        }
    }

    // Find JSON boundaries
    let start = content.find(|c: char| c == '{' || c == '[').unwrap_or(0);
    let end = content
        .rfind(|c: char| c == '}' || c == ']')
        .map(|i| i + 1)
        .unwrap_or(content.len());

    let json = content[start..end].trim().to_string();

    // Try to parse as-is first
    if serde_json::from_str::<serde_json::Value>(&json).is_ok() {
        return json;
    }

    // Fix malformed JSON
    let mut json = json;

    // Fix unquoted keys: `key:` → `"key":`
    let key_re = regex::Regex::new(r"(?m)^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*)(:)").unwrap();
    json = key_re.replace_all(&json, r#"$1"$2"$3"#).to_string();

    // Fix comma-before-closing: `,}` or `,]`
    let comma_re = regex::Regex::new(r",(\s*[\]}])").unwrap();
    json = comma_re.replace_all(&json, r#"$1"#).to_string();

    // Balance braces and brackets
    let mut braces: i32 = 0;
    let mut brackets: i32 = 0;
    for c in json.chars() {
        match c {
            '{' => braces += 1,
            '}' => braces = (braces - 1).max(0),
            '[' => brackets += 1,
            ']' => brackets = (brackets - 1).max(0),
            _ => {}
        }
    }

    if braces > 0 {
        json.push_str(&"}".repeat(braces as usize));
    }
    if brackets > 0 {
        json.push_str(&"]".repeat(brackets as usize));
    }

    json
}
