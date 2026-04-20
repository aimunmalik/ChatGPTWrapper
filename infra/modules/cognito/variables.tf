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
