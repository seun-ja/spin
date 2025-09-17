use spin_expressions::{provider::ProviderVariableKind, Key, Provider, ProviderResolver};
use spin_locked_app::Variable;

fn initialize_provider_resolver(
    variable_key: Option<String>,
    default: Option<String>,
    providers: Vec<Box<dyn Provider>>,
) -> anyhow::Result<ProviderResolver> {
    let mut provider_resolver = ProviderResolver::new(if let Some(key) = variable_key {
        vec![(
            key.to_string(),
            Variable {
                description: None,
                default,
                secret: false,
            },
        )]
    } else {
        vec![]
    })?;

    for provider in providers {
        provider_resolver.add_provider(provider as _);
    }

    Ok(provider_resolver)
}

struct StaticProvider {
    resolver: ProviderResolver,
}

impl StaticProvider {
    fn with_variables(
        variable_key: Option<String>,
        default: Option<String>,
        other_providers: Option<Box<dyn Provider>>,
    ) -> Self {
        let resolver = initialize_provider_resolver(
            variable_key,
            default,
            if other_providers.is_some() {
                vec![Box::new(StaticMockProvider), other_providers.unwrap()]
            } else {
                vec![Box::new(StaticMockProvider)]
            },
        )
        .unwrap();

        Self { resolver }
    }
}

struct DynamicProvider {
    resolver: ProviderResolver,
}

impl DynamicProvider {
    fn with_variables(
        variable_key: Option<String>,
        default: Option<String>,
        other_providers: Option<Box<dyn Provider>>,
    ) -> Self {
        let resolver = initialize_provider_resolver(
            variable_key,
            default,
            if other_providers.is_some() {
                vec![Box::new(DynamicMockProvider), other_providers.unwrap()]
            } else {
                vec![Box::new(DynamicMockProvider)]
            },
        )
        .unwrap();

        Self { resolver }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn single_static_provider_with_no_variable_provided_is_valid() -> anyhow::Result<()> {
    let static_provider = StaticProvider::with_variables(None, None, None);

    static_provider
        .resolver
        .validate_variable_existence()
        .await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn if_single_static_provider_has_variable_value_validation_succeeds() -> anyhow::Result<()> {
    let static_provider = StaticProvider::with_variables(Some("foo".to_string()), None, None);

    static_provider
        .resolver
        .validate_variable_existence()
        .await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn if_there_is_a_single_static_provider_and_it_does_not_contain_a_required_variable_then_validation_fails(
) -> anyhow::Result<()> {
    let static_provider = StaticProvider::with_variables(Some("bar".to_string()), None, None);

    assert!(static_provider
        .resolver
        .validate_variable_existence()
        .await
        .is_err());

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn if_there_is_a_dynamic_provider_then_validation_succeeds_even_if_a_static_provider_without_the_variable_is_in_play(
) -> anyhow::Result<()> {
    let dynamic_provider = DynamicProvider::with_variables(Some("baz".to_string()), None, None);

    dynamic_provider
        .resolver
        .validate_variable_existence()
        .await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn if_there_is_a_dynamic_provider_and_a_static_provider_then_validation_succeeds_even_if_a_static_provider_without_the_variable_is_in_play(
) -> anyhow::Result<()> {
    let dynamic_provider = DynamicProvider::with_variables(
        Some("baz".to_string()),
        None,
        Some(Box::new(StaticMockProvider)),
    );

    dynamic_provider
        .resolver
        .validate_variable_existence()
        .await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn if_there_is_a_dynamic_provider_and_a_static_provider_then_validation_succeeds_even_if_a_static_provider_with_the_variable_is_in_play(
) -> anyhow::Result<()> {
    let dynamic_provider = DynamicProvider::with_variables(
        Some("baz".to_string()),
        Some("foo".to_string()),
        Some(Box::new(StaticMockProvider)),
    );

    dynamic_provider
        .resolver
        .validate_variable_existence()
        .await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn static_provider_with_two_static_providers() -> anyhow::Result<()> {
    let static_provider = StaticProvider::with_variables(
        Some("bar".to_string()),
        Some("hay".to_string()),
        Some(Box::new(StaticMockProvider)),
    );

    static_provider
        .resolver
        .validate_variable_existence()
        .await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn static_provider_with_two_static_providers_where_first_provider_does_not_have_data_while_second_provider_does(
) -> anyhow::Result<()> {
    let static_provider = StaticProvider::with_variables(
        Some("bar".to_string()),
        None,
        Some(Box::new(ExtraStaticMockProvider)),
    );

    static_provider
        .resolver
        .validate_variable_existence()
        .await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn static_provider_with_two_static_providers_neither_has_data() -> anyhow::Result<()> {
    let static_provider = StaticProvider::with_variables(
        Some("hello".to_string()),
        None,
        Some(Box::new(ExtraStaticMockProvider)),
    );

    assert!(static_provider
        .resolver
        .validate_variable_existence()
        .await
        .is_err());

    Ok(())
}

#[derive(Debug)]
struct StaticMockProvider;

#[spin_world::async_trait]
impl Provider for StaticMockProvider {
    async fn get(&self, key: &Key) -> anyhow::Result<Option<String>> {
        match key.as_str() {
            "foo" => Ok(Some("bar".to_string())),
            _ => Ok(None),
        }
    }

    fn kind(&self) -> ProviderVariableKind {
        ProviderVariableKind::Static
    }
}

#[derive(Debug)]
struct ExtraStaticMockProvider;

#[spin_world::async_trait]
impl Provider for ExtraStaticMockProvider {
    async fn get(&self, key: &Key) -> anyhow::Result<Option<String>> {
        match key.as_str() {
            "bar" => Ok(Some("hey".to_string())),
            _ => Ok(None),
        }
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
