use std::fmt::Display;

use reqwest::{
    header::{HeaderMap, HeaderValue},
    Body, Client, Url,
};
use serde::Serialize;
use spin_world::v2::llm::{self as wasi_llm};

use crate::InferResponseBody;

pub(crate) struct OpenAIAgentEngine;

impl OpenAIAgentEngine {
    pub async fn infer(
        auth_token: &String,
        url: &Url,
        mut client: Option<Client>,
        model: wasi_llm::InferencingModel,
        prompt: String,
        params: wasi_llm::InferencingParams,
    ) -> Result<wasi_llm::InferencingResult, wasi_llm::Error> {
        let client = client.get_or_insert_with(Default::default);

        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("bearer {}", auth_token)).map_err(|_| {
                wasi_llm::Error::RuntimeError("Failed to create authorization header".to_string())
            })?,
        );
        spin_telemetry::inject_trace_context(&mut headers);

        let chat_url = url
            .join("/chat/completions")
            .map_err(|_| wasi_llm::Error::RuntimeError("Failed to create URL".to_string()))?;

        tracing::info!("Sending remote inference request to {chat_url}");

        let body = CreateChatCompletionRequest {
            messages: vec![Message {
                role: Role::User, // TODO: Joshua: make customizable
                content: prompt,
            }],
            model: model.into(),
            max_completion_tokens: todo!(),
            frequency_penalty: Some(params.repeat_penalty), // TODO: change to frequency_penalty
            reasoning_effort: todo!(),
            audio: todo!(),
            verbosity: todo!(),
        };

        let resp = client
            .request(reqwest::Method::POST, chat_url)
            .headers(headers)
            .body(body)
            .send()
            .await
            .map_err(|err| {
                wasi_llm::Error::RuntimeError(format!("POST /infer request error: {err}"))
            })?;

        match resp.json::<InferResponseBody>().await {
            Ok(val) => Ok(val.into()),
            Err(err) => Err(wasi_llm::Error::RuntimeError(format!(
                "Failed to deserialize response for \"POST  /index\": {err}"
            ))),
        }
    }

    pub async fn generate_embeddings(
        _auth_token: &str,
        _url: &Url,
        mut _client: Option<Client>,
        _model: wasi_llm::EmbeddingModel,
        _data: Vec<String>,
    ) -> Result<wasi_llm::EmbeddingsResult, wasi_llm::Error> {
        todo!("What's an embedding?")
    }
}

#[derive(Serialize, Debug)]
struct CreateChatCompletionRequest {
    messages: Vec<Message>,
    model: Model,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<Verbosity>,
}

impl From<CreateChatCompletionRequest> for Body {
    fn from(val: CreateChatCompletionRequest) -> Self {
        Body::from(serde_json::to_string(&val).unwrap())
    }
}

#[derive(Serialize, Debug)]
pub struct AudioOptions {
    pub voice: String,
    pub format: String,
}

#[derive(Serialize, Debug)]
enum Verbosity {
    Low,
    Medium,
    High,
}

#[derive(Serialize, Debug)]
enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
}

impl Display for ReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReasoningEffort::Minimal => write!(f, "minimal"),
            ReasoningEffort::Low => write!(f, "low"),
            ReasoningEffort::Medium => write!(f, "medium"),
            ReasoningEffort::High => write!(f, "high"),
        }
    }
}

#[derive(Serialize, Debug)]
enum InputType {
    Text,
    Audio,
    Image,
    Video,
}

impl Display for InputType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputType::Text => write!(f, "text"),
            InputType::Audio => write!(f, "audio"),
            InputType::Image => write!(f, "image"),
            InputType::Video => write!(f, "video"),
        }
    }
}

#[derive(Serialize, Debug)]
enum Model {
    GPT5,
    GPT5Mini,
    GPT5Nano,
    GPT5Chat,
    GPT45,
    GPT41,
    GPT41Mini,
    GPT41Nano,
    GPT4,
    GPT4o,
    GPT4oMini,
    O4Mini,
    O3,
    O1,
}

impl From<String> for Model {
    fn from(value: String) -> Self {
        match value.as_str() {
            "gpt-5" => Model::GPT5,
            "gpt-5-mini" => Model::GPT5Mini,
            "gpt-5-nano" => Model::GPT5Nano,
            "gpt-5-chat" => Model::GPT5Chat,
            "gpt-4.5" => Model::GPT45,
            "gpt-4.1" => Model::GPT41,
            "gpt-4.1-mini" => Model::GPT41Mini,
            "gpt-4.1-nano" => Model::GPT41Nano,
            "gpt-4" => Model::GPT4,
            "gpt-4o" => Model::GPT4o,
            "gpt-4o-mini" => Model::GPT4oMini,
            "o4-mini" => Model::O4Mini,
            "o3" => Model::O3,
            "o1" => Model::O1,
            _ => Model::GPT4,
        }
    }
}

impl Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::GPT5 => write!(f, "gpt-5"),
            Model::GPT5Mini => write!(f, "gpt-5-mini"),
            Model::GPT5Nano => write!(f, "gpt-5-nano"),
            Model::GPT5Chat => write!(f, "gpt-5-chat"),
            Model::GPT45 => write!(f, "gpt-4.5"),
            Model::GPT41 => write!(f, "gpt-4.1"),
            Model::GPT41Mini => write!(f, "gpt-4.1-mini"),
            Model::GPT41Nano => write!(f, "gpt-4.1-nano"),
            Model::GPT4 => write!(f, "gpt-4"),
            Model::GPT4o => write!(f, "gpt-4o"),
            Model::GPT4oMini => write!(f, "gpt-4o-mini"),
            Model::O4Mini => write!(f, "o4-mini"),
            Model::O3 => write!(f, "o3"),
            Model::O1 => write!(f, "o1"),
        }
    }
}

#[derive(Serialize, Debug)]
struct Message {
    role: Role,
    content: String,
}

#[derive(Serialize, Debug)]
enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}
