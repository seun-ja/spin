use std::fmt::Debug;

use async_trait::async_trait;
use serde::Deserialize;

use crate::Key;

/// A config provider.
#[async_trait]
pub trait Provider: Debug + Send + Sync {
    /// Returns the value at the given config path, if it exists.
    async fn get(&self, key: &Key) -> anyhow::Result<Option<String>>;
    fn kind(&self) -> &ProviderVariableKind;
}

/// The dynamism of a Provider.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub enum ProviderVariableKind {
    /// Variable must be declared on start
    #[serde(rename = "static")]
    Static,
    /// Variable can be made available at runtime
    #[serde(rename = "dynamic")]
    #[default]
    Dynamic,
}
