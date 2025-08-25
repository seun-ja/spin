use reqwest::{
    header::{HeaderMap, HeaderValue},
    Client, Url,
};
use serde::Serialize;
use spin_world::v2::llm::{self as wasi_llm};

use crate::{
    schema::{EmbeddingModels, EncodingFormat, Model, Prompt, ResponseError, Role},
    CreateChatCompletionResponse, CreateEmbeddingResponse,
};

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
            .join("/v1/chat/completions")
            .map_err(|_| wasi_llm::Error::RuntimeError("Failed to create URL".to_string()))?;

        tracing::info!("Sending remote inference request to {chat_url}");

        let body = CreateChatCompletionRequest {
            // TODO: Make Role customizable
            messages: vec![Prompt::new(Role::User, prompt)],
            model: model.as_str().try_into()?,
            max_completion_tokens: Some(params.max_tokens),
            frequency_penalty: Some(params.repeat_penalty),
            reasoning_effort: None,
            verbosity: None,
        };

        let resp = client
            .request(reqwest::Method::POST, chat_url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|err| {
                wasi_llm::Error::RuntimeError(format!("POST /infer request error: {err}"))
            })?;

        match resp.json::<CreateChatCompletionResponses>().await {
            Ok(CreateChatCompletionResponses::Success(val)) => Ok(val.into()),
            Ok(CreateChatCompletionResponses::Error { error }) => Err(error.into()),
            Err(err) => Err(wasi_llm::Error::RuntimeError(format!(
                "Failed to deserialize response for \"POST  /index\": {err}"
            ))),
        }
    }

    pub async fn generate_embeddings(
        auth_token: &str,
        url: &Url,
        mut client: Option<Client>,
        model: wasi_llm::EmbeddingModel,
        data: Vec<String>,
    ) -> Result<wasi_llm::EmbeddingsResult, wasi_llm::Error> {
        let client = client.get_or_insert_with(Default::default);

        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("bearer {}", auth_token)).map_err(|_| {
                wasi_llm::Error::RuntimeError("Failed to create authorization header".to_string())
            })?,
        );
        spin_telemetry::inject_trace_context(&mut headers);

        let body = CreateEmbeddingRequest {
            input: data,
            model: model.as_str().try_into()?,
            encoding_format: None,
            dimensions: None,
            user: None,
        };

        let resp = client
            .request(
                reqwest::Method::POST,
                url.join("/v1/embeddings").map_err(|_| {
                    wasi_llm::Error::RuntimeError("Failed to create URL".to_string())
                })?,
            )
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|err| {
                wasi_llm::Error::RuntimeError(format!("POST /embed request error: {err}"))
            })?;

        match resp.json::<CreateEmbeddingResponses>().await {
            Ok(CreateEmbeddingResponses::Success(val)) => Ok(val.into()),
            Ok(CreateEmbeddingResponses::Error { error }) => Err(error.into()),
            Err(err) => Err(wasi_llm::Error::RuntimeError(format!(
                "Failed to deserialize response  for \"POST  /embed\": {err}"
            ))),
        }
    }
}

#[derive(Serialize, Debug)]
struct CreateChatCompletionRequest {
    messages: Vec<Prompt>,
    model: Model,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verbosity: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct CreateEmbeddingRequest {
    input: Vec<String>,
    model: EmbeddingModels,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<EncodingFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum CreateChatCompletionResponses {
    Success(CreateChatCompletionResponse),
    Error { error: ResponseError },
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum CreateEmbeddingResponses {
    Success(CreateEmbeddingResponse),
    Error { error: ResponseError },
}
