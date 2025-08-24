use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Deserialize)]
struct CodestralMessage {
    role: String,
    content: String,
    tool_calls: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct CodestralChoice {
    index: i32,
    finish_reason: String,
    message: CodestralMessage,
}

#[derive(Debug, Deserialize)]
struct CodestralUsage {
    prompt_tokens: i32,
    total_tokens: i32,
    completion_tokens: i32,
}

#[derive(Debug, Deserialize)]
struct CodestralResponse {
    id: String,
    created: i64,
    model: String,
    usage: CodestralUsage,
    object: String,
    choices: Vec<CodestralChoice>,
}

#[derive(Debug, Serialize)]
struct CodestralRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix: Option<String>,
    stop: Vec<String>,
    max_tokens: i32,
    temperature: f64,
}

/// Make the actual API request to Codestral and return completion text
fn make_codestral_request(
    api_key: &str,
    prompt: &str,
    suffix: Option<&str>,
    max_tokens: i32,
    stops: Vec<String>,
) -> Result<String, Box<dyn std::error::Error>> {
    let payload = CodestralRequest {
        model: "codestral-latest".to_string(),
        prompt: prompt.to_string(),
        suffix: suffix.and_then(|s| if s.is_empty() { None } else { Some(s.to_string()) }),
        stop: stops,
        max_tokens,
        temperature: 0.0,
    };

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_millis(5000))  // 5s timeout for autosuggestions
        .build()?;
    
    let response = client
        .post("https://codestral.mistral.ai/v1/fim/completions")
        .header("Content-Type", "application/json")
        .header("Accept", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&payload)
        .send()?;

    if !response.status().is_success() {
        let error_text = response.text()?;
        eprintln!("Error response: {}", error_text);
        return Err(format!("HTTP error: {}", error_text).into());
    }

    // Parse the response
    let codestral_response: CodestralResponse = response.json()?;
    
    // Extract completion text from the first choice
    if !codestral_response.choices.is_empty() {
        Ok(codestral_response.choices[0].message.content.clone())
    } else {
        Ok(String::new())
    }
}

/// Get code completion from Mistral Codestral
///
/// # Arguments
/// * `prompt` - The code context/prompt to complete
/// * `api_key` - Codestral API key (or None to use CODESTRAL_API_KEY env var)
/// * `max_tokens` - Maximum tokens to generate (default: 128)
/// * `stops` - List of stop sequences (default: ["\n\n"])
/// * `suffix` - Code that comes after the cursor (default: "")
pub fn codestral_autocomplete(
    prompt: &str,
    api_key: Option<&str>,
    max_tokens: Option<i32>,
    stops: Option<Vec<String>>,
    suffix: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
    // Get API key from parameter or environment
    let api_key = if let Some(key) = api_key {
        key.to_string()
    } else {
        env::var("CODESTRAL_API_KEY").map_err(|_| {
            "CODESTRAL_API_KEY environment variable not set and no API key provided"
        })?
    };

    // Set defaults
    let stops = stops.unwrap_or_else(|| vec!["\n\n".to_string()]);
    let max_tokens = max_tokens.unwrap_or(128);

    make_codestral_request(
        &api_key,
        prompt,
        suffix,
        max_tokens,
        stops,
    )
}

/// A client that caches the Codestral API key
pub struct CodestralClient {
    api_key: String,
}

impl CodestralClient {
    /// Create a new CodestralClient
    /// 
    /// # Arguments
    /// * `api_key` - Optional API key. If None, will use CODESTRAL_API_KEY env var
    pub fn new(api_key: Option<String>) -> Result<Self, Box<dyn std::error::Error>> {
        let api_key = if let Some(key) = api_key {
            key
        } else {
            env::var("CODESTRAL_API_KEY").map_err(|_| {
                "CODESTRAL_API_KEY environment variable not set and no API key provided"
            })?
        };

        Ok(CodestralClient { api_key })
    }

    /// Get code completion from Codestral
    pub fn autocomplete(
        &self,
        prompt: &str,
        max_tokens: Option<i32>,
        stops: Option<Vec<String>>,
        suffix: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Use defaults
        let stops = stops.unwrap_or_else(|| vec!["\n\n".to_string()]);
        let max_tokens = max_tokens.unwrap_or(128);

        make_codestral_request(
            &self.api_key,
            prompt,
            suffix,
            max_tokens,
            stops,
        )
    }
}

// For backward compatibility, keep the old function name as an alias
pub fn copilot_autocomplete(
    prompt: &str,
    _github_token: &str,  // Ignored, will use CODESTRAL_API_KEY
    max_tokens: Option<i32>,
    _temperature: Option<f64>,  // Ignored, always 0
    stops: Option<Vec<String>>,
    _language: Option<&str>,  // Ignored
    suffix: Option<&str>,
    _is_copilot_token: bool,  // Ignored
) -> Result<String, Box<dyn std::error::Error>> {
    codestral_autocomplete(prompt, None, max_tokens, stops, suffix)
}

// For backward compatibility, keep CopilotClient as an alias
pub type CopilotClient = CodestralClient;

// Main function for testing
fn main() {
    // Get API key from environment
    let api_key = env::var("CODESTRAL_API_KEY").unwrap_or_else(|_| {
        eprintln!("Error: Set CODESTRAL_API_KEY environment variable");
        std::process::exit(1);
    });

    // Simple example - ffmpeg command completion
    let prompt = ">  # ffmpeg concat all images (#.png) in current folder into mp4";

    println!("Testing Codestral completion with prompt:");
    println!("{}", prompt);
    println!();

    match codestral_autocomplete(
        prompt,
        Some(&api_key),
        Some(128),
        Some(vec!["\n\n".to_string()]),
        None,
    ) {
        Ok(completion) => {
            println!("Completion:");
            println!("{}", completion);
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    // Example with CodestralClient for multiple requests
    println!("\n--- Testing CodestralClient ---");
    
    match CodestralClient::new(Some(api_key)) {
        Ok(client) => {
            let prompts = vec![
                "def fibonacci(n):\n    \"\"\"Calculate nth Fibonacci number\"\"\"",
                "fn bubble_sort(arr: &mut [i32]) {\n    // Sort array in place",
                "class Calculator:\n    def __init__(self):",
            ];

            for (i, prompt) in prompts.iter().enumerate() {
                println!("\nPrompt {}:", i + 1);
                println!("{}", prompt);
                match client.autocomplete(prompt, None, None, None) {
                    Ok(completion) => {
                        println!("Completion:");
                        println!("{}", completion);
                    }
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
        }
        Err(e) => eprintln!("Error creating client: {}", e),
    }
}