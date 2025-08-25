use serde::{Deserialize, Serialize};
use spin_world::v2::llm as wasi_llm;

/// LLM model
#[derive(Serialize, Debug)]
pub enum Model {
    #[serde(rename = "gpt-5")]
    GPT5,
    #[serde(rename = "gpt-5-mini")]
    GPT5Mini,
    #[serde(rename = "gpt-5-nano")]
    GPT5Nano,
    #[serde(rename = "gpt-5-chat")]
    GPT5Chat,
    #[serde(rename = "gpt-4.5")]
    GPT45,
    #[serde(rename = "gpt-4.1")]
    GPT41,
    #[serde(rename = "gpt-4.1-mini")]
    GPT41Mini,
    #[serde(rename = "gpt-4.1-nano")]
    GPT41Nano,
    #[serde(rename = "gpt-4")]
    GPT4,
    #[serde(rename = "gpt-4o")]
    GPT4o,
    #[serde(rename = "gpt-4o-mini")]
    GPT4oMini,
    #[serde(rename = "o4-mini")]
    O4Mini,
    #[serde(rename = "o3")]
    O3,
    #[serde(rename = "o1")]
    O1,
}

impl TryFrom<&str> for Model {
    type Error = wasi_llm::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "gpt-5" => Ok(Model::GPT5),
            "gpt-5-mini" => Ok(Model::GPT5Mini),
            "gpt-5-nano" => Ok(Model::GPT5Nano),
            "gpt-5-chat" => Ok(Model::GPT5Chat),
            "gpt-4.5" => Ok(Model::GPT45),
            "gpt-4.1" => Ok(Model::GPT41),
            "gpt-4.1-mini" => Ok(Model::GPT41Mini),
            "gpt-4.1-nano" => Ok(Model::GPT41Nano),
            "gpt-4" => Ok(Model::GPT4),
            "gpt-4o" => Ok(Model::GPT4o),
            "gpt-4o-mini" => Ok(Model::GPT4oMini),
            "o4-mini" => Ok(Model::O4Mini),
            "o3" => Ok(Model::O3),
            "o1" => Ok(Model::O1),
            _ => Err(wasi_llm::Error::InvalidInput(format!(
                "{value} is not a valid model name" // TODO: Joshua: Have some public docs to state the supported models to point users to
            ))),
        }
    }
}

#[derive(Serialize, Debug)]
pub struct Prompt {
    role: Role,
    content: String,
}

impl Prompt {
    pub fn new(role: Role, content: String) -> Self {
        Self { role, content }
    }
}

#[derive(Serialize, Debug)]
pub enum Role {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "tool")]
    Tool,
}

impl TryFrom<&str> for Role {
    type Error = wasi_llm::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "system" => Ok(Role::System),
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            "tool" => Ok(Role::Tool),
            _ => Err(wasi_llm::Error::InvalidInput(format!(
                "{value} not a valid role"
            ))),
        }
    }
}

#[derive(Serialize, Debug)]
pub enum EncodingFormat {
    #[serde(rename = "float")]
    Float,
    #[serde(rename = "base64")]
    Base64,
}

impl TryFrom<&str> for EncodingFormat {
    type Error = wasi_llm::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "float" => Ok(EncodingFormat::Float),
            "base64" => Ok(EncodingFormat::Base64),
            _ => Err(wasi_llm::Error::InvalidInput(format!(
                "{value} not a valid encoding format"
            ))),
        }
    }
}

#[derive(Serialize, Debug)]
pub enum EmbeddingModels {
    #[serde(rename = "text-embedding-ada-002")]
    TextEmbeddingAda002,
    #[serde(rename = "text-embedding-3-small")]
    TextEmbedding3Small,
    #[serde(rename = "text-embedding-3-large")]
    TextEmbedding3Large,
    Custom(String),
}

impl TryFrom<&str> for EmbeddingModels {
    type Error = wasi_llm::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "text-embedding-ada-002" => Ok(EmbeddingModels::TextEmbeddingAda002),
            "text-embedding-3-small" => Ok(EmbeddingModels::TextEmbedding3Small),
            "text-embedding-3-large" => Ok(EmbeddingModels::TextEmbedding3Large),
            _ => Ok(EmbeddingModels::Custom(value.to_string())),
        }
    }
}

#[derive(Serialize, Debug)]
enum ReasoningEffort {
    #[serde(rename = "minimal")]
    Minimal,
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "high")]
    High,
}

impl TryFrom<&str> for ReasoningEffort {
    type Error = wasi_llm::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "minimal" => Ok(ReasoningEffort::Minimal),
            "low" => Ok(ReasoningEffort::Low),
            "medium" => Ok(ReasoningEffort::Medium),
            "high" => Ok(ReasoningEffort::High),
            _ => Err(wasi_llm::Error::InvalidInput(format!(
                "{value} not a recognized reasoning effort",
            ))),
        }
    }
}

#[derive(Serialize, Debug)]
enum Verbosity {
    Low,
    Medium,
    High,
}

impl TryFrom<&str> for Verbosity {
    type Error = wasi_llm::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "low" => Ok(Verbosity::Low),
            "medium" => Ok(Verbosity::Medium),
            "high" => Ok(Verbosity::High),
            _ => Err(wasi_llm::Error::InvalidInput(format!(
                "{value} not a recognized verbosity",
            ))),
        }
    }
}

#[derive(Deserialize)]
pub struct ChatCompletionChoice {
    #[serde(rename = "index")]
    /// The index of the choice in the list of choices
    _index: u32,
    pub message: ChatCompletionResponseMessage,
    /// The reason the model stopped generating tokens. This will be `stop` if the model hit a
    /// natural stop point or a provided stop sequence,
    #[serde(rename = "finish_reason")]
    _finish_reason: String,
    /// Log probability information for the choice.
    #[serde(rename = "logprobs")]
    _logprobs: Option<Logprobs>,
}

#[derive(Deserialize)]
/// A chat completion message generated by the model.
pub struct ChatCompletionResponseMessage {
    /// The role of the author of this message
    #[serde(rename = "role")]
    _role: String,
    /// The contents of the message
    pub content: String,
    /// The refusal message generated by the model
    #[serde(rename = "refusal")]
    _refusal: Option<String>,
}

#[derive(Deserialize)]
pub struct Logprobs {
    /// A list of message content tokens with log probability information.
    #[serde(rename = "content")]
    _content: Option<Vec<String>>,
    /// A list of message refusal tokens with log probability information.
    #[serde(rename = "refusal")]
    _refusal: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub struct Embedding {
    /// The index of the embedding in the list of embeddings..
    #[serde(rename = "index")]
    _index: u32,
    /// The embedding vector, which is a list of floats. The length of vector depends on the model as
    /// listed in the [embedding guide](https://platform.openai.com/docs/guides/embeddings).
    pub embedding: Vec<f32>,
    /// The object type, which is always "embedding"
    #[serde(rename = "object")]
    _object: String,
}

#[derive(Deserialize, Default)]
pub struct ResponseError {
    pub message: String,
    #[serde(rename = "type")]
    _t: String,
    #[serde(rename = "param")]
    _param: Option<String>,
    #[serde(rename = "code")]
    _code: String,
}

impl From<ResponseError> for wasi_llm::Error {
    fn from(value: ResponseError) -> Self {
        wasi_llm::Error::RuntimeError(value.message)
    }
}
