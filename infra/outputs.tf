output "openai_endpoint" {
  value = azurerm_cognitive_account.openai.endpoint
}

output "search_endpoint" {
  value = "https://${azurerm_search_service.search.name}.search.windows.net"
}

output "chat_deployment" {
  value = azurerm_cognitive_deployment.chat.name
}

output "embed_deployment" {
  value = azurerm_cognitive_deployment.embed.name
}

# `make env-from-tf` writes this into .env.azure (git-ignored). Endpoints and names only: nothing secret,
# because access is granted to the logged-in identity through RBAC, not through keys.
output "dotenv" {
  value = <<-EOT
    AZURE_REGION=${azurerm_search_service.search.location}
    AZURE_OPENAI_ENDPOINT=${azurerm_cognitive_account.openai.endpoint}
    AZURE_OPENAI_EMBED_DEPLOYMENT=${azurerm_cognitive_deployment.embed.name}
    AZURE_CHAT_DEPLOYMENT=${azurerm_cognitive_deployment.chat.name}
    AZURE_SEARCH_ENDPOINT=https://${azurerm_search_service.search.name}.search.windows.net
  EOT
}
