variable "aws_region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "env" {
  description = "Environment name. Used as a resource prefix."
  type        = string
  default     = "staging"
}

variable "vpc_cidr" {
  description = "CIDR block for the staging VPC. Distinct from dev (10.20.0.0/20) so the two environments never collide if peered."
  type        = string
  default     = "10.30.0.0/20"
}

variable "flow_logs_enabled" {
  description = "Whether VPC flow logs are enabled. Usually false in lower environments to save cost; always true in prod."
  type        = bool
  default     = false
}

variable "log_retention_days" {
  description = "CloudWatch Logs retention in days."
  type        = number
  default     = 30
}

variable "cognito_domain_suffix" {
  description = "Suffix for the Cognito hosted-UI domain. Must be globally unique in the region. Pick something short and random-ish, e.g. 'anna-staging-42'. No default — supplied at deploy time via TF_VAR_cognito_domain_suffix (the STAGING_COGNITO_DOMAIN_SUFFIX GitHub variable) so it can't collide with dev's."
  type        = string
}

variable "cognito_callback_urls" {
  description = "OAuth callback URLs for the SPA. Localhost-only until the CloudFront default domain exists."
  type        = list(string)
  # After first apply, add the CloudFront default domain (see cloudfront_url
  # output) here and re-apply, since it's unknown until the distribution exists.
  default = [
    "http://localhost:5173/callback",
  ]
}

variable "cognito_logout_urls" {
  description = "Sign-out redirect URLs for the SPA. Cognito requires EXACT match — query string and path included — so any logout_uri the SPA sends must appear in this list verbatim. Localhost-only until the CloudFront default domain exists."
  type        = list(string)
  # After first apply, add the CloudFront default domain (see cloudfront_url
  # output) here and re-apply, since it's unknown until the distribution exists.
  default = [
    "http://localhost:5173",
    # /signed-out.html is a STATIC page outside the SPA. buildLogoutUrl
    # sends users here after Cognito clears the session. The static page
    # has no JavaScript / AuthProvider, so the user's still-alive session
    # can't silently re-authenticate them.
    "http://localhost:5173/signed-out.html",
  ]
}

# ──────────────────────────────────────────────────────────────────────────
# Microsoft Entra (M365) federation — see docs/SETUP_M365_SSO.md
#
# Staging deliberately runs WITHOUT Microsoft SSO: all three values default
# to "" so federation is off and the pool is local Cognito users only
# (username/password + TOTP). The deploy-staging workflow passes NO entra
# TF_VARs, so these defaults stand.
# ──────────────────────────────────────────────────────────────────────────

variable "entra_tenant_id" {
  description = "Microsoft Entra tenant (directory) ID. Empty in staging (federation off)."
  type        = string
  default     = ""
}

variable "entra_client_id" {
  description = "Application (client) ID of the Entra app registration. Empty in staging (federation off)."
  type        = string
  default     = ""
}

variable "entra_client_secret" {
  description = "Client secret value from the Entra app registration. Sensitive. Empty in staging (federation off)."
  type        = string
  default     = ""
  sensitive   = true
}

variable "cors_allow_origins" {
  description = "Origins allowed by API Gateway CORS. Localhost-only until the CloudFront default domain exists."
  type        = list(string)
  # After first apply, add the CloudFront default domain (see cloudfront_url
  # output) here and re-apply, since it's unknown until the distribution exists.
  default = [
    "http://localhost:5173",
  ]
}

variable "bedrock_model_id" {
  description = "Bedrock model ID the chat Lambda invokes. Cross-region inference profile IDs are prefixed with 'us.'."
  type        = string
  default     = "us.anthropic.claude-sonnet-4-6"
}

variable "message_ttl_days" {
  description = "How long to keep messages in DynamoDB before TTL expires them."
  type        = number
  default     = 90
}

variable "cloudfront_price_class" {
  description = "CloudFront price class. PriceClass_100 (US/EU) is cheapest."
  type        = string
  default     = "PriceClass_100"
}

variable "waf_rate_limit" {
  description = "WAF rate limit per IP per 5-minute window."
  type        = number
  default     = 2000
}

variable "alarm_email" {
  description = "Email subscribed to the alarms SNS topic. AWS sends a one-time confirmation link — click it once after first apply."
  type        = string
  default     = "aimun@annaautismcare.com"
}

variable "bedrock_monthly_budget_usd" {
  description = "Monthly Bedrock spend cap in USD. Warnings at 50/80/100% email you; at 120% a kill-switch IAM deny policy auto-attaches to the chat Lambda role."
  type        = number
  default     = 100
}

variable "github_org" {
  description = "GitHub org/owner for the OIDC trust policy."
  type        = string
  default     = "aimunmalik"
}

variable "github_repo" {
  description = "GitHub repo name for the OIDC trust policy."
  type        = string
  default     = "ChatGPTWrapper"
}
