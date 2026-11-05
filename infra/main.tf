# Everything step 19 needs on Azure: Azure OpenAI (chat + embeddings) and Azure AI Search (Free), keyless.
# `make infra-apply` creates it, `make infra-destroy` removes it after the eval: nothing keeps running (or billing).

data "azurerm_client_config" "current" {}

locals {
  # AI Search and the OpenAI subdomain are global DNS names: a short hash of the subscription makes them unique
  # without a random provider, and stays the same between runs
  suffix = substr(sha1("${data.azurerm_client_config.current.subscription_id}-${var.prefix}"), 0, 6)
  name   = "${var.prefix}-${local.suffix}"
  tags   = { project = "19-azure-ai-rag", managed_by = "terraform" }
  me     = data.azurerm_client_config.current.object_id # whoever ran `az login`
}

resource "azurerm_resource_group" "rag" {
  name     = "rg-${local.name}"
  location = var.location
  tags     = local.tags
}

# --- Azure OpenAI -----------------------------------------------------------------------------------------------

resource "azurerm_cognitive_account" "openai" {
  name                = "aoai-${local.name}"
  location            = coalesce(var.openai_location, var.location)
  resource_group_name = azurerm_resource_group.rag.name
  kind                = "OpenAI"
  sku_name            = "S0"
  # Entra ID tokens only work against a custom subdomain (https://<name>.openai.azure.com)
  custom_subdomain_name = "aoai-${local.name}"
  # no API keys at all: access is the RBAC role below, used through `az login` / DefaultAzureCredential
  local_auth_enabled            = false
  public_network_access_enabled = true
  tags                          = local.tags
}

resource "azurerm_cognitive_deployment" "chat" {
  name                 = var.chat_model # the deployment name = LLM_MODEL in .env.azure
  cognitive_account_id = azurerm_cognitive_account.openai.id
  model {
    format  = "OpenAI"
    name    = var.chat_model
    version = var.chat_model_version
  }
  sku {
    name     = var.deployment_sku
    capacity = var.chat_capacity_ktpm
  }
  version_upgrade_option = "NoAutoUpgrade" # the model must not change between the eval runs
}

resource "azurerm_cognitive_deployment" "embed" {
  name                 = var.embed_model
  cognitive_account_id = azurerm_cognitive_account.openai.id
  model {
    format  = "OpenAI"
    name    = var.embed_model
    version = var.embed_model_version
  }
  sku {
    name     = var.deployment_sku
    capacity = var.embed_capacity_ktpm
  }
  version_upgrade_option = "NoAutoUpgrade"
  # one deployment operation at a time per account: parallel ones fail with a conflict
  depends_on = [azurerm_cognitive_deployment.chat]
}

# --- Azure AI Search (Free tier) --------------------------------------------------------------------------------

resource "azurerm_search_service" "search" {
  name                = "srch-${local.name}"
  location            = coalesce(var.search_location, var.location)
  resource_group_name = azurerm_resource_group.rag.name
  # free: 50 MB, shared infrastructure, no semantic ranker, no replicas/partitions; one per subscription
  sku = "free"
  # data-plane RBAC instead of admin/query keys (works on every tier, including free)
  local_authentication_enabled = false
  tags                         = local.tags
}

# --- Who may do what (least privilege for the person running the eval) ------------------------------------------

resource "azurerm_role_assignment" "openai_user" {
  scope                = azurerm_cognitive_account.openai.id
  role_definition_name = "Cognitive Services OpenAI User" # call deployments, nothing else
  principal_id         = local.me
}

resource "azurerm_role_assignment" "search_service" {
  scope                = azurerm_search_service.search.id
  role_definition_name = "Search Service Contributor" # create / delete the index schema
  principal_id         = local.me
}

resource "azurerm_role_assignment" "search_data" {
  scope                = azurerm_search_service.search.id
  role_definition_name = "Search Index Data Contributor" # upload documents and query them
  principal_id         = local.me
}

# --- Budget alert ----------------------------------------------------------------------------------------------

resource "azurerm_consumption_budget_subscription" "rag" {
  count           = var.budget_amount > 0 ? 1 : 0
  name            = "budget-${local.name}"
  subscription_id = "/subscriptions/${data.azurerm_client_config.current.subscription_id}"
  amount          = var.budget_amount
  time_grain      = "Monthly"

  time_period {
    start_date = var.budget_start_date
  }

  notification {
    enabled        = true
    operator       = "GreaterThan"
    threshold      = 50
    threshold_type = "Actual"
    contact_emails = [var.budget_contact_email]
  }

  notification {
    enabled        = true
    operator       = "GreaterThan"
    threshold      = 100
    threshold_type = "Forecasted"
    contact_emails = [var.budget_contact_email]
  }

  lifecycle {
    precondition {
      condition     = var.budget_contact_email != ""
      error_message = "Set budget_contact_email in terraform.tfvars, or budget_amount = 0 to skip the budget."
    }
  }
}
