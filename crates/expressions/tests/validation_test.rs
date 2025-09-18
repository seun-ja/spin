use std::collections::HashMap;

use spin_expressions::{provider::ProviderVariableKind, Key, Provider, ProviderResolver};
use spin_locked_app::Variable;

#[derive(Default)]
struct ResolverTester {
    providers: Vec<Box<dyn Provider>>,
}

impl ResolverTester {
    fn new() -> Self {
        Self::default()
    }

    fn with_provider(mut self, provider: Box<dyn Provider>) -> Self {
        self.providers.push(provider);
        self
    }

    fn make_resolver(
        self,
        key: Option<&str>,
        default: Option<&str>,
    ) -> anyhow::Result<ProviderResolver> {
        let mut provider_resolver = ProviderResolver::new(
            key.map(|k| {
                vec![(
                    k.to_string(),
                    Variable {
                        description: None,
                        default: default.map(ToString::to_string),
                        secret: false,
                    },
                )]
            })
            .unwrap_or_default(),
        )?;

        for provider in self.providers {
            provider_resolver.add_provider(provider as _);
        }

        Ok(provider_resolver)
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn single_static_provider_with_no_variable_provided_is_valid() -> anyhow::Result<()> {
    let resolver = ResolverTester::new()
        .with_provider(Box::new(StaticMockProvider::with_variables("foo", "bar")))
        .make_resolver(None, None)?;

    resolver.validate_variables().await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn if_single_static_provider_has_variable_value_validation_succeeds() -> anyhow::Result<()> {
    let resolver = ResolverTester::new()
        .with_provider(Box::new(StaticMockProvider::with_variables("foo", "bar")))
        .make_resolver(Some("foo"), None)?;

    resolver.validate_variables().await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn if_there_is_a_single_static_provider_and_it_does_not_contain_a_required_variable_then_validation_fails(
) -> anyhow::Result<()> {
    let resolver = ResolverTester::new()
        .with_provider(Box::new(StaticMockProvider::with_variables("foo", "bar")))
        .make_resolver(Some("baz"), None)?;

    assert!(resolver.validate_variables().await.is_err());

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn if_there_is_a_dynamic_provider_then_validation_succeeds_even_if_a_static_provider_without_the_variable_is_in_play(
) -> anyhow::Result<()> {
    let resolver = ResolverTester::new()
        .with_provider(Box::new(DynamicMockProvider))
        .make_resolver(Some("baz"), None)?;

    resolver.validate_variables().await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn if_there_is_a_dynamic_provider_and_a_static_provider_then_validation_succeeds_even_if_a_static_provider_without_the_variable_is_in_play(
) -> anyhow::Result<()> {
    let resolver = ResolverTester::new()
        .with_provider(Box::new(DynamicMockProvider))
        .with_provider(Box::new(StaticMockProvider::with_variables("foo", "bar")))
        .make_resolver(Some("baz"), None)?;

    resolver.validate_variables().await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn if_there_is_a_dynamic_provider_and_a_static_provider_then_validation_succeeds_even_if_a_static_provider_with_the_variable_is_in_play(
) -> anyhow::Result<()> {
    let resolver = ResolverTester::new()
        .with_provider(Box::new(DynamicMockProvider))
        .with_provider(Box::new(StaticMockProvider::with_variables("foo", "bar")))
        .make_resolver(Some("baz"), Some("coo"))?;

    resolver.validate_variables().await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn static_provider_with_two_static_providers() -> anyhow::Result<()> {
    let resolver = ResolverTester::new()
        .with_provider(Box::new(StaticMockProvider::with_variables("foo", "bar")))
        .with_provider(Box::new(StaticMockProvider::with_variables("baz", "hay")))
        .make_resolver(Some("baz"), Some("hay"))?;

    resolver.validate_variables().await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn static_provider_with_two_static_providers_where_first_provider_does_not_have_data_while_second_provider_does(
) -> anyhow::Result<()> {
    let resolver = ResolverTester::new()
        .with_provider(Box::new(StaticMockProvider::with_variables("foo", "bar")))
        .with_provider(Box::new(StaticMockProvider::with_variables("baz", "hay")))
        .make_resolver(Some("baz"), None)?;

    resolver.validate_variables().await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn static_provider_with_two_static_providers_neither_has_data_is_invalid(
) -> anyhow::Result<()> {
    let resolver = ResolverTester::new()
        .with_provider(Box::new(StaticMockProvider::with_variables("foo", "bar")))
        .with_provider(Box::new(StaticMockProvider::with_variables("baz", "hay")))
        .make_resolver(Some("hello"), None)?;

    assert!(resolver.validate_variables().await.is_err());

    Ok(())
}

#[derive(Debug)]
struct StaticMockProvider {
    variables: HashMap<String, Option<String>>,
}

impl StaticMockProvider {
    fn with_variables(key: &str, value: &str) -> Self {
        Self {
            variables: HashMap::from([(key.into(), Some(value.into()))]),
        }
    }
}

#[spin_world::async_trait]
impl Provider for StaticMockProvider {
    async fn get(&self, key: &Key) -> anyhow::Result<Option<String>> {
        Ok(self.variables.get(key.as_str()).cloned().flatten())
    }

    fn kind(&self) -> ProviderVariableKind {
        ProviderVariableKind::Static
    }
}

#[derive(Debug)]
struct DynamicMockProvider;

#[spin_world::async_trait]
impl Provider for DynamicMockProvider {
    async fn get(&self, _key: &Key) -> anyhow::Result<Option<String>> {
        Ok(None)
    }

    fn kind(&self) -> ProviderVariableKind {
        ProviderVariableKind::Dynamic
    }
}
