use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Deserialize)]
struct TokenResponse {
    token: String,
    expires_at: f64,
}

#[derive(Debug, Deserialize)]
struct CompletionChoice {
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CompletionResponse {
    choices: Option<Vec<CompletionChoice>>,
}

#[derive(Debug, Serialize)]
struct CompletionExtra {
    language: String,
    next_indent: i32,
    trim_by_indentation: bool,
}

#[derive(Debug, Serialize)]
struct CompletionRequest {
    prompt: String,
    suffix: String,
    max_tokens: i32,
    temperature: f64,
    top_p: f64,
    n: i32,
    stop: Vec<String>,
    stream: bool,
    extra: CompletionExtra,
}

/// Exchange GitHub token for Copilot token
fn get_copilot_token(github_token: &str) -> Result<(String, f64), Box<dyn std::error::Error>> {
    let auth_header = if github_token.starts_with("ghu_") {
        format!("Bearer {}", github_token)
    } else {
        format!("token {}", github_token)
    };

    let client = reqwest::blocking::Client::new();
    let mut headers = HashMap::new();
    headers.insert("content-type", "application/json");
    headers.insert("accept", "application/json");
    headers.insert("User-Agent", "GitHubCopilotChat/0.12.2024062801");
    headers.insert("Editor-Version", "vscode/1.93.1");
    headers.insert("Editor-Plugin-Version", "copilot-chat/0.12.2024062801");
    headers.insert("VScode-SessionId", "12345678-1234-1234-1234-123456789012");

    let response = client
        .get("https://api.github.com/copilot_internal/v2/token")
        .header("Authorization", auth_header)
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .header("User-Agent", "GitHubCopilotChat/0.12.2024062801")
        .header("Editor-Version", "vscode/1.93.1")
        .header("Editor-Plugin-Version", "copilot-chat/0.12.2024062801")
        .header("VScode-SessionId", "12345678-1234-1234-1234-123456789012")
        .send()?;

    if !response.status().is_success() {
        let error_text = response.text()?;
        eprintln!("Error response: {}", error_text);
        return Err(format!("HTTP error: {}", error_text).into());
    }

    let result: TokenResponse = response.json()?;
    Ok((result.token, result.expires_at))
}

/// Calculate temperature based on prompt line count
fn calculate_temperature(prompt: &str) -> f64 {
    let line_count = prompt.lines().count().max(1) - 2;
    let line_count = line_count.max(1);

    if line_count <= 1 {
        0.0
    } else if line_count <= 10 {
        0.2
    } else if line_count < 20 {
        0.4
    } else {
        0.8
    }
}

/// Make the actual API request to Copilot and return completion text
fn make_copilot_request(
    copilot_token: &str,
    prompt: &str,
    suffix: &str,
    max_tokens: i32,
    temperature: f64,
    stops: Vec<String>,
    language: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let payload = CompletionRequest {
        prompt: prompt.to_string(),
        suffix: suffix.to_string(),
        max_tokens,
        temperature,
        top_p: 1.0,
        n: 1,
        stop: stops,
        stream: true,
        extra: CompletionExtra {
            language: language.to_string(),
            next_indent: 0,
            trim_by_indentation: true,
        },
    };

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_millis(5000))  // 5s timeout for autosuggestions
        .build()?;
    
    let response = client
        .post("https://copilot-proxy.githubusercontent.com/v1/engines/copilot-codex/completions")
        .header("OpenAI-Intent", "copilot-ghost")
        .header("OpenAI-Organization", "github-copilot")
        .header("Authorization", format!("Bearer {}", copilot_token))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()?;

    if !response.status().is_success() {
        return Err(format!("HTTP error: {}", response.status()).into());
    }

    // Parse streaming response
    let mut completion_text = String::new();

    // Read the entire response text at once
    let response_text = response.text()?;

    // Process each line in the response
    for line in response_text.lines() {
        if line.starts_with("data: ") {
            let data_str = &line[6..];
            if data_str == "[DONE]" {
                break;
            }
            if let Ok(data) = serde_json::from_str::<CompletionResponse>(data_str) {
                if let Some(choices) = data.choices {
                    if !choices.is_empty() {
                        if let Some(text) = &choices[0].text {
                            completion_text.push_str(text);
                        }
                    }
                }
            }
        }
    }

    Ok(completion_text)
}

/// Get code completion from GitHub Copilot
///
/// # Arguments
/// * `prompt` - The code context/prompt to complete
/// * `github_token` - GitHub personal access token OR Copilot token
/// * `max_tokens` - Maximum tokens to generate (default: 200)
/// * `temperature` - Sampling temperature 0-1 (default: auto based on line count)
/// * `stops` - List of stop sequences (default: ["\n\n"])
/// * `language` - Programming language (default: "python")
/// * `suffix` - Code that comes after the cursor (default: "")
/// * `is_copilot_token` - If true, treat token as Copilot token (skip exchange)
pub fn copilot_autocomplete(
    prompt: &str,
    github_token: &str,
    max_tokens: Option<i32>,
    temperature: Option<f64>,
    stops: Option<Vec<String>>,
    language: Option<&str>,
    suffix: Option<&str>,
    is_copilot_token: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    // Get Copilot token
    let copilot_token = if is_copilot_token || github_token.starts_with("gho_") {
        github_token.to_string()
    } else {
        let (token, _) = get_copilot_token(github_token)?;
        token
    };

    // Set defaults
    let temperature = temperature.unwrap_or_else(|| calculate_temperature(prompt));
    let stops = stops.unwrap_or_else(|| vec!["\n\n".to_string()]);
    let max_tokens = max_tokens.unwrap_or(200);
    let language = language.unwrap_or("python");
    let suffix = suffix.unwrap_or("");

    make_copilot_request(
        &copilot_token,
        prompt,
        suffix,
        max_tokens,
        temperature,
        stops,
        language,
    )
}

/// A client that caches the Copilot token to avoid repeated token exchanges
pub struct CopilotClient {
    github_token: String,
    is_copilot_token: bool,
    copilot_token: Option<String>,
    token_expiry: f64,
}

impl CopilotClient {
    /// Create a new CopilotClient
    pub fn new(github_token: String, is_copilot_token: Option<bool>) -> Self {
        let is_copilot_token = is_copilot_token.unwrap_or_else(|| github_token.starts_with("gho_"));
        let (copilot_token, token_expiry) = if is_copilot_token {
            (Some(github_token.clone()), f64::INFINITY)
        } else {
            (None, 0.0)
        };

        CopilotClient {
            github_token,
            is_copilot_token,
            copilot_token,
            token_expiry,
        }
    }

    /// Get Copilot token, refreshing if expired
    fn get_copilot_token(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        if self.is_copilot_token {
            return Ok(self.github_token.clone());
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as f64;

        if self.copilot_token.is_none() || now >= self.token_expiry {
            let (token, expiry) = get_copilot_token(&self.github_token)?;
            self.copilot_token = Some(token.clone());
            self.token_expiry = expiry;
            Ok(token)
        } else {
            Ok(self.copilot_token.as_ref().unwrap().clone())
        }
    }

    /// Get code completion from GitHub Copilot with cached token
    pub fn autocomplete(
        &mut self,
        prompt: &str,
        max_tokens: Option<i32>,
        temperature: Option<f64>,
        stops: Option<Vec<String>>,
        language: Option<&str>,
        suffix: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let copilot_token = self.get_copilot_token()?;

        // Use defaults
        let temperature = temperature.unwrap_or_else(|| calculate_temperature(prompt));
        let stops = stops.unwrap_or_else(|| vec!["\n\n".to_string()]);
        let max_tokens = max_tokens.unwrap_or(200);
        let language = language.unwrap_or("python");
        let suffix = suffix.unwrap_or("");

        make_copilot_request(
            &copilot_token,
            prompt,
            suffix,
            max_tokens,
            temperature,
            stops,
            language,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_temperature() {
        assert_eq!(calculate_temperature("single line"), 0.0);
        assert_eq!(calculate_temperature("line1\nline2\nline3"), 0.0);
        assert_eq!(calculate_temperature("line1\nline2\nline3\nline4\nline5"), 0.2);

        let many_lines = (0..15).map(|i| format!("line{}", i)).collect::<Vec<_>>().join("\n");
        assert_eq!(calculate_temperature(&many_lines), 0.4);

        let very_many_lines = (0..25).map(|i| format!("line{}", i)).collect::<Vec<_>>().join("\n");
        assert_eq!(calculate_temperature(&very_many_lines), 0.8);
    }

    #[test]
    fn test_copilot_client_creation() {
        let client = CopilotClient::new("ghu_test123".to_string(), None);
        assert!(!client.is_copilot_token);

        let client = CopilotClient::new("gho_test123".to_string(), None);
        assert!(client.is_copilot_token);

        let client = CopilotClient::new("test123".to_string(), Some(true));
        assert!(client.is_copilot_token);
    }
}

// Main function for testing
fn main() {
    // Get token from environment
    let github_token = env::var("GITHUB_TOKEN")
        .or_else(|_| env::var("GITHUB_COPILOT_ACCESS_TOKEN"))
        .unwrap_or_else(|_| {
            eprintln!("Error: Set GITHUB_TOKEN or GITHUB_COPILOT_ACCESS_TOKEN environment variable");
            std::process::exit(1);
        });

    // Simple example
    let prompt = "def fibonacci(n):\n    \"\"\"Calculate nth Fibonacci number\"\"\"";

    match copilot_autocomplete(
        prompt,
        &github_token,
        None,
        None,
        None,
        None,
        None,
        false,
    ) {
        Ok(completion) => println!("Completion: {}", completion),
        Err(e) => eprintln!("Error: {}", e),
    }

    // Example with CopilotClient for multiple requests
    println!("\n--- Testing CopilotClient ---");
    let mut client = CopilotClient::new(github_token, None);

    let prompts = vec![
        "def factorial(n):\n    \"\"\"Calculate factorial of n\"\"\"",
        "fn bubble_sort(arr: &mut [i32]) {\n    // Sort array in place",
        "class Calculator:\n    def __init__(self):",
    ];

    for (i, prompt) in prompts.iter().enumerate() {
        println!("\nPrompt {}:", i + 1);
        println!("{}", prompt);
        match client.autocomplete(prompt, None, None, None, None, None) {
            Ok(completion) => println!("Completion: {}", completion),
            Err(e) => eprintln!("Error: {}", e),
        }
    }
}
