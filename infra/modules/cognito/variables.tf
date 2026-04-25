variable "env" {
  description = "Environment name (dev, prod)."
  type        = string
}

variable "domain_suffix" {
  description = "Suffix appended to the Cognito hosted-UI domain prefix. Must be globally unique within the region."
  type        = string
}

variable "callback_urls" {
  description = "OAuth callback URLs for the SPA client. Use http://localhost:5173/callback for local dev."
  type        = list(string)
}

variable "logout_urls" {
  description = "Sign-out redirect URLs for the SPA client."
  type        = list(string)
}

variable "deletion_protection" {
  description = "Whether deletion protection is enabled on the user pool. Should be true in prod."
  type        = bool
  default     = true
}

variable "advanced_security_mode" {
  description = "Cognito advanced security mode: OFF, AUDIT, or ENFORCED. Production should use ENFORCED."
  type        = string
  default     = "ENFORCED"

  validation {
    condition     = contains(["OFF", "AUDIT", "ENFORCED"], var.advanced_security_mode)
    error_message = "Must be OFF, AUDIT, or ENFORCED."
  }
}

variable "tags" {
  description = "Tags applied to all resources in the module."
  type        = map(string)
  default     = {}
}

# ──────────────────────────────────────────────────────────────────────────
# Microsoft Entra ID (Azure AD / Microsoft 365) federation
#
# When all three of these are non-empty, the module wires Entra in as a
# federated OIDC IdP and adds it to the SPA client's allowed providers.
# Leave any one blank to skip federation entirely — the user pool stays
# username/password + TOTP only.
#
# The client secret is sensitive: store it in a gitignored tfvars file or
# pass via TF_VAR_entra_client_secret from GitHub Actions Variables.
# ──────────────────────────────────────────────────────────────────────────

variable "entra_tenant_id" {
  description = "Microsoft Entra ID tenant (directory) ID. Empty = federation disabled."
  type        = string
  default     = ""
}

variable "entra_client_id" {
  description = "Application (client) ID of the Entra app registration. Empty = federation disabled."
  type        = string
  default     = ""
}

variable "entra_client_secret" {
  description = "Client secret value from the Entra app registration. Sensitive."
  type        = string
  default     = ""
  sensitive   = true
}

variable "entra_provider_name" {
  description = "Identity provider name shown on the Hosted UI button. Letters/numbers/underscore only."
  type        = string
  default     = "Microsoft"
}
