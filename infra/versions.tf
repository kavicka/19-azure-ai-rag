terraform {
  required_version = ">= 1.6"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 5.12"
    }
  }
}

# The subscription comes from ARM_SUBSCRIPTION_ID (the Makefile sets it from `az account show`), and the identity
# from the Azure CLI login: no credentials in this repo.
provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false # `make infra-destroy` removes everything this step created
    }
  }
  # register only what this step uses instead of the provider's long default list (a fresh Students
  # subscription would otherwise spend minutes registering providers it never needs)
  resource_provider_registrations = "none"
  resource_providers_to_register  = ["Microsoft.CognitiveServices", "Microsoft.Search"]
}
