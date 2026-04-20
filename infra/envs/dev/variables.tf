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
  description = "OAuth callback URLs for the SPA. Defaults cover local dev + the CloudFront domain."
  type        = list(string)
  default = [
    "http://localhost:5173/callback",
    "https://dr8xfgmss2sy0.cloudfront.net/callback",
  ]
}

variable "cognito_logout_urls" {
  description = "Sign-out redirect URLs for the SPA."
  type        = list(string)
  default = [
    "http://localhost:5173",
    "https://dr8xfgmss2sy0.cloudfront.net",
  ]
}

variable "cors_allow_origins" {
  description = "Origins allowed by API Gateway CORS. Includes local dev + the CloudFront domain."
  type        = list(string)
  default = [
    "http://localhost:5173",
    "https://dr8xfgmss2sy0.cloudfront.net",
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
