locals {
  name_prefix = "anna-chat-${var.env}"
}

resource "aws_cognito_user_pool" "this" {
  name = "${local.name_prefix}-users"

  deletion_protection = var.deletion_protection ? "ACTIVE" : "INACTIVE"

  mfa_configuration = "ON"

  software_token_mfa_configuration {
    enabled = true
  }

  password_policy {
    minimum_length                   = 12
    require_lowercase                = true
    require_uppercase                = true
    require_numbers                  = true
    require_symbols                  = true
    temporary_password_validity_days = 3
  }

  admin_create_user_config {
    allow_admin_create_user_only = true

    invite_message_template {
      email_subject = "Welcome to Praxis · by ANNA"
      email_message = "Hello,\n\nYou have been invited to Praxis, ANNA's clinical intelligence platform.\n\nUsername: {username}\nTemporary password: {####}\n\nSign in at the URL provided by your ANNA administrator. You'll be asked to set a new password and enroll an authenticator app on first sign-in.\n\nPraxis handles protected health information under ANNA's HIPAA Business Associate Agreement — do not share your credentials."
      sms_message   = "Praxis by ANNA — username {username}, temp password {####}"
    }
  }

  username_attributes = ["email"]

  auto_verified_attributes = ["email"]

  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }

  email_configuration {
    email_sending_account = "COGNITO_DEFAULT"
  }

  schema {
    name                     = "email"
    attribute_data_type      = "String"
    mutable                  = true
    required                 = true
    developer_only_attribute = false

    string_attribute_constraints {
      min_length = 3
      max_length = 254
    }
  }

  schema {
    name                     = "name"
    attribute_data_type      = "String"
    mutable                  = true
    required                 = true
    developer_only_attribute = false

    string_attribute_constraints {
      min_length = 1
      max_length = 128
    }
  }

  user_pool_add_ons {
    advanced_security_mode = var.advanced_security_mode
  }

  tags = var.tags
}

resource "aws_cognito_user_pool_domain" "this" {
  domain       = "${local.name_prefix}-${var.domain_suffix}"
  user_pool_id = aws_cognito_user_pool.this.id
}

resource "aws_cognito_user_pool_client" "spa" {
  name         = "${local.name_prefix}-spa"
  user_pool_id = aws_cognito_user_pool.this.id

  generate_secret = false

  allowed_oauth_flows                  = ["code"]
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_scopes                 = ["openid", "email", "profile"]

  callback_urls = var.callback_urls
  logout_urls   = var.logout_urls

  supported_identity_providers = ["COGNITO"]

  explicit_auth_flows = [
    "ALLOW_USER_SRP_AUTH",
    "ALLOW_REFRESH_TOKEN_AUTH",
  ]

  access_token_validity  = 60
  id_token_validity      = 60
  refresh_token_validity = 30

  token_validity_units {
    access_token  = "minutes"
    id_token      = "minutes"
    refresh_token = "days"
  }

  prevent_user_existence_errors = "ENABLED"
  enable_token_revocation       = true

  auth_session_validity = 3
}

resource "aws_cognito_user_group" "admins" {
  name         = "admins"
  user_pool_id = aws_cognito_user_pool.this.id
  description  = "ANNA Chat administrators. Can invite users."
  precedence   = 1
}

resource "aws_cognito_user_group" "users" {
  name         = "users"
  user_pool_id = aws_cognito_user_pool.this.id
  description  = "Standard ANNA Chat users (clinicians)."
  precedence   = 10
}
