variable "env" {
  description = "Environment name (dev, prod)."
  type        = string
}

variable "kms_key_arn" {
  description = "Reserved for future use (attachment bucket, etc.). SPA bucket itself uses AES256 since contents are already-public SPA assets."
  type        = string
  default     = null
}

variable "price_class" {
  description = "CloudFront price class. PriceClass_100 = US/EU only (cheapest). PriceClass_All = worldwide."
  type        = string
  default     = "PriceClass_100"
}

variable "rate_limit_per_5min" {
  description = "WAF rate limit per IP per 5-minute window."
  type        = number
  default     = 2000
}

variable "csp_connect_extra" {
  description = "Extra origins allowed in the CSP connect-src directive. Provide Cognito + API Gateway URLs."
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags applied to all resources in the module."
  type        = map(string)
  default     = {}
}
