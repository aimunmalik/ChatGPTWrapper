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
  description = "OAuth callback URLs for the SPA. In dev, typically http://localhost:5173/callback. Add the CloudFront URL once deployed."
  type        = list(string)
  default     = ["http://localhost:5173/callback"]
}

variable "cognito_logout_urls" {
  description = "Sign-out redirect URLs for the SPA."
  type        = list(string)
  default     = ["http://localhost:5173"]
}
