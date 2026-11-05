variable "prefix" {
  description = "Short name used in every resource name (a hash of the subscription is appended for uniqueness)."
  type        = string
  default     = "rag19"
  validation {
    condition     = can(regex("^[a-z][a-z0-9]{2,11}$", var.prefix))
    error_message = "3-12 lowercase letters/digits, starting with a letter (it ends up in DNS names)."
  }
}

variable "location" {
  description = "Region for the resource group and Azure AI Search. Azure for Students allows only some regions (Azure Policy → Assignments → 'Allowed resource deployment regions')."
  type        = string
  default     = "germanywestcentral"
}

variable "openai_location" {
  description = "Region for Azure OpenAI, if the model or quota is only available elsewhere. Empty = var.location."
  type        = string
  default     = ""
}

variable "search_location" {
  description = "Region for Azure AI Search, if the Free tier has no capacity in var.location (InsufficientResourcesAvailable). Empty = var.location."
  type        = string
  default     = ""
}

variable "deployment_sku" {
  description = "GlobalStandard (cheapest, may process in any region), DataZoneStandard (stays in the EU, ~10 % more). Regional 'Standard' is retired for gpt-4o-mini."
  type        = string
  default     = "GlobalStandard"
  validation {
    condition     = contains(["GlobalStandard", "DataZoneStandard", "Standard"], var.deployment_sku)
    error_message = "One of GlobalStandard, DataZoneStandard, Standard."
  }
}

variable "chat_model" {
  description = "Chat model for generation (the cheapest 'mini' model the subscription has quota for)."
  type        = string
  default     = "gpt-4.1-mini"
}

variable "chat_model_version" {
  description = "Pinned model version, so the measured model cannot change under the eval."
  type        = string
  default     = "2025-04-14"
}

variable "chat_capacity_ktpm" {
  description = "Chat deployment capacity in thousands of tokens per minute; it also sets requests per minute (10 = 10 RPM, which throttles a sequential eval). Costs nothing by itself (billing is per token). Must fit the subscription's quota (az cognitiveservices usage list)."
  type        = number
  default     = 100
  validation {
    condition     = var.chat_capacity_ktpm >= 1
    error_message = "At least 1 (= 1,000 tokens per minute)."
  }
}

variable "embed_model" {
  type    = string
  default = "text-embedding-3-small"
}

variable "embed_model_version" {
  type    = string
  default = "1"
}

variable "embed_capacity_ktpm" {
  description = "Embedding deployment capacity in thousands of tokens per minute. Indexing ~1.2M tokens at 50k TPM takes ~25 min."
  type        = number
  default     = 50
}

variable "budget_amount" {
  description = "Monthly budget alert in the subscription currency. 0 = no budget (Azure for Students still has its hard credit limit)."
  type        = number
  default     = 5
}

variable "budget_contact_email" {
  description = "Where budget alerts go. Required when budget_amount > 0. Keep it in terraform.tfvars (git-ignored)."
  type        = string
  default     = ""
}

variable "budget_start_date" {
  description = "First day of the current month, RFC3339 (Azure requires the first of a month), e.g. 2026-11-01T00:00:00Z."
  type        = string
  default     = "2026-11-01T00:00:00Z"
}
