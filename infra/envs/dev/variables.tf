variable "aws_region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "env" {
  description = "Environment name. Used as a resource prefix."
  type        = string
  default     = "dev"
}

variable "vpc_cidr" {
  description = "CIDR block for the dev VPC."
  type        = string
  default     = "10.20.0.0/20"
}

variable "flow_logs_enabled" {
  description = "Whether VPC flow logs are enabled. Usually false in dev to save cost; always true in prod."
  type        = bool
  default     = false
}

variable "log_retention_days" {
  description = "CloudWatch Logs retention in days."
  type        = number
  default     = 30
}

variable "cognito_domain_suffix" {
  description = "Suffix for the Cognito hosted-UI domain. Must be globally unique in the region. Pick something short and random-ish, e.g. 'anna42'."
  type        = string
}

variable "cognito_callback_urls" {
  description = "OAuth callback URLs for the SPA. Defaults cover local dev, CloudFront default domain, and the praxis.annaautismcare.com custom domain."
  type        = list(string)
  default = [
    "http://localhost:5173/callback",
    "https://dr8xfgmss2sy0.cloudfront.net/callback",
    "https://praxis.annaautismcare.com/callback",
  ]
}

variable "cognito_logout_urls" {
  description = "Sign-out redirect URLs for the SPA. Cognito requires EXACT match — query string and path included — so any logout_uri the SPA sends must appear in this list verbatim."
  type        = list(string)
  default = [
    "http://localhost:5173",
    "https://dr8xfgmss2sy0.cloudfront.net",
    "https://praxis.annaautismcare.com",
    # /login?signedout=1 is what buildLogoutUrl now sends so the SPA can
    # show a "you're signed out" page instead of immediately re-initiating
    # the sign-in flow (which silently re-auths via SSO and makes sign-out
    # appear to do nothing).
    "http://localhost:5173/login?signedout=1",
    "https://dr8xfgmss2sy0.cloudfront.net/login?signedout=1",
    "https://praxis.annaautismcare.com/login?signedout=1",
  ]
}

# ──────────────────────────────────────────────────────────────────────────
# Microsoft Entra (M365) federation — see docs/SETUP_M365_SSO.md
#
# All three values come from the Entra app registration. Tenant + client
# IDs are not secret (visible to anyone with M365 admin); the client
# secret is. Keep entra_client_secret in terraform.tfvars (gitignored)
# OR set TF_VAR_entra_client_secret as a GitHub Actions Variable so it's
# only ever in CI memory, never on disk.
#
# Leaving any one of these empty disables the federation entirely.
# ──────────────────────────────────────────────────────────────────────────

variable "entra_tenant_id" {
  description = "Microsoft Entra tenant (directory) ID, e.g. 11111111-2222-3333-4444-555555555555."
  type        = string
  default     = ""
}

variable "entra_client_id" {
  description = "Application (client) ID of the Entra app registration."
  type        = string
  default     = ""
}

variable "entra_client_secret" {
  description = "Client secret value from the Entra app registration. Sensitive."
  type        = string
  default     = ""
  sensitive   = true
}

variable "cors_allow_origins" {
  description = "Origins allowed by API Gateway CORS. Includes local dev, CloudFront default domain, and the praxis.annaautismcare.com custom domain."
  type        = list(string)
  default = [
    "http://localhost:5173",
    "https://dr8xfgmss2sy0.cloudfront.net",
    "https://praxis.annaautismcare.com",
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
