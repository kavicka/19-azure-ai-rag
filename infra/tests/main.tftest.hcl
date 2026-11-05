# `terraform test` with a mocked azurerm provider: checks the plan's security and cost settings without an Azure
# login (runs in CI). Run: terraform init -backend=false && terraform test

mock_provider "azurerm" {
  mock_data "azurerm_client_config" {
    defaults = {
      subscription_id = "00000000-0000-0000-0000-000000000001"
      object_id       = "11111111-1111-1111-1111-111111111111"
      tenant_id       = "22222222-2222-2222-2222-222222222222"
    }
  }
}

variables {
  budget_contact_email = "alerts@example.com"
}

run "keyless_and_free" {
  command = plan

  assert {
    condition     = azurerm_cognitive_account.openai.local_auth_enabled == false
    error_message = "Azure OpenAI must not accept API keys"
  }
  assert {
    condition     = azurerm_search_service.search.local_authentication_enabled == false
    error_message = "Azure AI Search must not accept admin/query keys"
  }
  assert {
    condition     = azurerm_search_service.search.sku == "free"
    error_message = "Search must stay on the Free tier (0 €)"
  }
  assert {
    condition     = azurerm_cognitive_account.openai.custom_subdomain_name == azurerm_cognitive_account.openai.name
    error_message = "Entra ID auth needs a custom subdomain"
  }
}

run "least_privilege_roles_for_the_caller" {
  command = plan

  assert {
    condition = toset([
      azurerm_role_assignment.openai_user.role_definition_name,
      azurerm_role_assignment.search_service.role_definition_name,
      azurerm_role_assignment.search_data.role_definition_name,
      ]) == toset([
      "Cognitive Services OpenAI User", "Search Service Contributor", "Search Index Data Contributor",
    ])
    error_message = "Unexpected role set"
  }
  assert {
    condition     = azurerm_role_assignment.openai_user.principal_id == "11111111-1111-1111-1111-111111111111"
    error_message = "Roles must go to the logged-in identity"
  }
}

run "pinned_models" {
  command = plan

  assert {
    condition     = azurerm_cognitive_deployment.chat.version_upgrade_option == "NoAutoUpgrade"
    error_message = "The measured model must not auto-upgrade"
  }
  assert {
    condition     = azurerm_cognitive_deployment.chat.model[0].version == "2025-04-14"
    error_message = "Chat model version must be pinned"
  }
  assert {
    condition     = azurerm_cognitive_deployment.embed.sku[0].name == "GlobalStandard"
    error_message = "Default SKU is GlobalStandard"
  }
}

run "names_are_dns_safe_and_stable" {
  command = plan

  assert {
    condition     = can(regex("^srch-rag19-[0-9a-f]{6}$", azurerm_search_service.search.name))
    error_message = "Search name must be prefix + 6-char hash"
  }
}

run "budget_on_by_default" {
  command = plan

  assert {
    condition     = length(azurerm_consumption_budget_subscription.rag) == 1
    error_message = "A budget alert is created by default"
  }
}

run "budget_off" {
  command = plan
  variables {
    budget_amount        = 0
    budget_contact_email = ""
  }

  assert {
    condition     = length(azurerm_consumption_budget_subscription.rag) == 0
    error_message = "budget_amount = 0 skips the budget"
  }
}

run "budget_without_email_fails" {
  command = plan
  variables {
    budget_contact_email = ""
  }
  expect_failures = [azurerm_consumption_budget_subscription.rag]
}
