use spin_expressions::{provider::ProviderVariableKind, Key, Provider};
use spin_factor_variables::{runtime_config::RuntimeConfig, VariablesFactor};
use spin_factors::{anyhow, RuntimeFactors};
use spin_factors_test::{toml, TestEnvironment};
use spin_world::v2::variables::Host;

#[derive(RuntimeFactors)]
struct TestFactors {
    variables: VariablesFactor,
}

#[tokio::test(flavor = "multi_thread")]
async fn provider_works() -> anyhow::Result<()> {
    let factors = TestFactors {
        variables: VariablesFactor::default(),
    };
    let providers = vec![Box::new(MockProvider) as _];
    let runtime_config = TestFactorsRuntimeConfig {
        variables: Some(RuntimeConfig { providers }),
    };
    let env = TestEnvironment::new(factors)
        .extend_manifest(toml! {
            [variables]
            foo = { required = true }

            [component.test-component]
            source = "does-not-exist.wasm"
            variables = { baz = "<{{ foo }}>" }
        })
        .runtime_config(runtime_config)?;

    let mut state = env.build_instance_state().await?;
    let val = state.variables.get("baz".into()).await?;
    assert_eq!(val, "<bar>");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn validate_variable_existence_successful() -> anyhow::Result<()> {
    let factors = TestFactors {
        variables: VariablesFactor::default(),
    };
    let providers = vec![Box::new(MockProvider) as _];
    let runtime_config = TestFactorsRuntimeConfig {
        variables: Some(RuntimeConfig { providers }),
    };
    let env = TestEnvironment::new(factors)
        .extend_manifest(toml! {
            [variables]
            foo = { required = true }

            [component.test-component]
            source = "does-not-exist.wasm"
        })
        .runtime_config(runtime_config)?;

    let state = env.build_instance_state().await?;
    let resolver = state.variables.expression_resolver();

    resolver.validate_variable_existence().await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn validate_variable_existence_successful_with_default_value() -> anyhow::Result<()> {
    let factors = TestFactors {
        variables: VariablesFactor::default(),
    };
    let providers = vec![Box::new(MockProvider) as _];
    let runtime_config = TestFactorsRuntimeConfig {
        variables: Some(RuntimeConfig { providers }),
    };
    let env = TestEnvironment::new(factors)
        .extend_manifest(toml! {
            [variables]
            baz = { default = "var" }

            [component.test-component]
            source = "does-not-exist.wasm"
        })
        .runtime_config(runtime_config)?;

    let state = env.build_instance_state().await?;
    let resolver = state.variables.expression_resolver();

    resolver.validate_variable_existence().await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn validate_variable_existence_successful_with_dynamic_provider() -> anyhow::Result<()> {
    let factors = TestFactors {
        variables: VariablesFactor::default(),
    };
    let providers = vec![Box::new(DynamicMockProvider) as _];
    let runtime_config = TestFactorsRuntimeConfig {
        variables: Some(RuntimeConfig { providers }),
    };
    let env = TestEnvironment::new(factors)
        .extend_manifest(toml! {
            [variables]
            baz = { required = true }

            [component.test-component]
            source = "does-not-exist.wasm"
        })
        .runtime_config(runtime_config)?;

    let state = env.build_instance_state().await?;
    let resolver = state.variables.expression_resolver();

    resolver.validate_variable_existence().await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn validate_variable_existence_fails() -> anyhow::Result<()> {
    let factors = TestFactors {
        variables: VariablesFactor::default(),
    };
    let providers = vec![Box::new(MockProvider) as _];
    let runtime_config = TestFactorsRuntimeConfig {
        variables: Some(RuntimeConfig { providers }),
    };
    let env = TestEnvironment::new(factors)
        .extend_manifest(toml! {
            [variables]
            baz = { required = true }

            [component.test-component]
            source = "does-not-exist.wasm"
        })
        .runtime_config(runtime_config)?;

    let state = env.build_instance_state().await?;
    let resolver = state.variables.expression_resolver();

    assert!(resolver.validate_variable_existence().await.is_err());

    Ok(())
}

#[derive(Debug)]
struct MockProvider;

#[spin_world::async_trait]
impl Provider for MockProvider {
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
