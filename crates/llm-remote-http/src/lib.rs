use anyhow::Result;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use spin_world::v2::llm::{self as wasi_llm};

use crate::{default::DefaultAgentEngine, open_ai::OpenAIAgentEngine};

mod default;
mod open_ai;

#[derive(Clone)]
pub enum Agent {
    //TODO: Joshua: Naming??!
    Default {
        auth_token: String,
        url: Url,
        client: Option<Client>,
    },
    OpenAI {
        auth_token: String,
        url: Url,
        client: Option<Client>,
    },
}

impl Agent {
    pub fn from(url: Url, auth_token: String, agent: Option<String>) -> Self {
        match agent {
            Some(agent_name) if agent_name == *"open_ai" => Agent::OpenAI {
                auth_token,
                url,
                client: None,
            },
            _ => Agent::Default {
                auth_token,
                url,
                client: None,
            },
        }
    }
}

#[derive(Clone)]
pub struct RemoteHttpLlmEngine {
    agent: Agent,
}

#[derive(Serialize)]
#[serde(rename_all(serialize = "camelCase"))]
struct InferRequestBodyParams {
    max_tokens: u32,
    repeat_penalty: f32,
    repeat_penalty_last_n_token_count: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
}

#[derive(Deserialize)]
#[serde(rename_all(deserialize = "camelCase"))]
struct InferUsage {
    prompt_token_count: u32,
    generated_token_count: u32,
}

#[derive(Deserialize)]
struct InferResponseBody {
    text: String,
    usage: InferUsage,
}

#[derive(Deserialize)]
#[serde(rename_all(deserialize = "camelCase"))]
struct EmbeddingUsage {
    prompt_token_count: u32,
}

#[derive(Deserialize)]
struct EmbeddingResponseBody {
    embeddings: Vec<Vec<f32>>,
    usage: EmbeddingUsage,
}

impl RemoteHttpLlmEngine {
    pub fn new(url: Url, auth_token: String, agent: Option<String>) -> Self {
        RemoteHttpLlmEngine {
            agent: Agent::from(url, auth_token, agent),
        }
    }

    pub async fn infer(
        &mut self,
        model: wasi_llm::InferencingModel,
        prompt: String,
        params: wasi_llm::InferencingParams,
    ) -> Result<wasi_llm::InferencingResult, wasi_llm::Error> {
        match &self.agent {
            Agent::OpenAI {
                auth_token,
                url,
                client,
            } => {
                OpenAIAgentEngine::infer(auth_token, url, client.clone(), model, prompt, params)
                    .await
            }
            Agent::Default {
                auth_token,
                url,
                client,
            } => {
                DefaultAgentEngine::infer(auth_token, url, client.clone(), model, prompt, params)
                    .await
            }
        }
    }

    pub async fn generate_embeddings(
        &mut self,
        model: wasi_llm::EmbeddingModel,
        data: Vec<String>,
    ) -> Result<wasi_llm::EmbeddingsResult, wasi_llm::Error> {
        match &self.agent {
            Agent::OpenAI {
                auth_token,
                url,
                client,
            } => {
                OpenAIAgentEngine::generate_embeddings(auth_token, url, client.clone(), model, data)
                    .await
            }
            Agent::Default {
                auth_token,
                url,
                client,
            } => {
                DefaultAgentEngine::generate_embeddings(
                    auth_token,
                    url,
                    client.clone(),
                    model,
                    data,
                )
                .await
            }
        }
    }

    pub fn url(&self) -> Url {
        match &self.agent {
            Agent::OpenAI { url, .. } => url.clone(),
            Agent::Default { url, .. } => url.clone(),
        }
    }
}

impl From<InferResponseBody> for wasi_llm::InferencingResult {
    fn from(value: InferResponseBody) -> Self {
        Self {
            text: value.text,
            usage: wasi_llm::InferencingUsage {
                prompt_token_count: value.usage.prompt_token_count,
                generated_token_count: value.usage.generated_token_count,
            },
        }
    }
}

impl From<EmbeddingResponseBody> for wasi_llm::EmbeddingsResult {
    fn from(value: EmbeddingResponseBody) -> Self {
        Self {
            embeddings: value.embeddings,
            usage: wasi_llm::EmbeddingsUsage {
                prompt_token_count: value.usage.prompt_token_count,
            },
        }
    }
}
