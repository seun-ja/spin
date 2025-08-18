use std::path::Path;

use anyhow::Result;
use spin_core::async_trait;
use spin_world::v2::llm as wasi_llm;

use crate::InferencingModel;

#[derive(Clone)]
pub(crate) struct OpenAIModels {}

impl OpenAIModels {
    pub async fn new(_model_dir: &Path) -> Result<Self> {
        Ok(Self {})
    }
}

#[async_trait]
impl InferencingModel for OpenAIModels {
    async fn infer(
        &self,
        _prompt: String,
        _params: wasi_llm::InferencingParams,
    ) -> anyhow::Result<wasi_llm::InferencingResult> {
        todo!()
    }
}
