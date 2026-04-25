locals {
  name_prefix = "anna-chat-${var.env}"

  # Entra federation is gated on all three credentials being present. We
  # avoid creating an OIDC IdP with placeholder values — Cognito accepts
  # them silently and then breaks on first sign-in attempt.
  entra_enabled = (
    var.entra_tenant_id != ""
    && var.entra_client_id != ""
    && var.entra_client_secret != ""
  )

  # When Entra is enabled, the SPA client allows both COGNITO (local
  # username/password — the break-glass path) and the federated provider.
  # When disabled, only COGNITO. Order matters for the Hosted UI: COGNITO
  # last so the federated button is visually primary.
  supported_idps = local.entra_enabled ? [var.entra_provider_name, "COGNITO"] : ["COGNITO"]
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

# ──────────────────────────────────────────────────────────────────────────
# Microsoft Entra ID federation (created only when entra_* vars are set)
#
# Cognito acts as the OAuth client to Entra: user clicks "Sign in with
# Microsoft" on the Hosted UI → bounce to login.microsoftonline.com →
# user authenticates against ANNA's M365 tenant (with whatever MFA Entra
# Conditional Access enforces) → bounce back to Cognito with an OIDC code
# → Cognito exchanges code for tokens, provisions / refreshes a federated
# user, and issues its own JWT to our SPA.
#
# OIDC discovery URL Cognito reads under the hood:
#   https://login.microsoftonline.com/{tenant}/v2.0/.well-known/openid-configuration
# Endpoints (authorize, token, jwks, userinfo) are auto-discovered from there.
#
# Attribute mapping: Entra issues `email` and `name` claims when the
# `email` and `profile` scopes are granted in the app registration AND
# the user has those values populated in M365. If `email` ever shows up
# blank on a federated sign-in, check the app registration's "Optional
# claims" blade and add `email` to the ID token.
# ──────────────────────────────────────────────────────────────────────────

resource "aws_cognito_identity_provider" "entra" {
  count         = local.entra_enabled ? 1 : 0
  user_pool_id  = aws_cognito_user_pool.this.id
  provider_name = var.entra_provider_name
  provider_type = "OIDC"

  provider_details = {
    client_id                 = var.entra_client_id
    client_secret             = var.entra_client_secret
    attributes_request_method = "GET"
    oidc_issuer               = "https://login.microsoftonline.com/${var.entra_tenant_id}/v2.0"
    authorize_scopes          = "openid profile email"
  }

  attribute_mapping = {
    email    = "email"
    name     = "name"
    username = "sub"
  }
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

  supported_identity_providers = local.supported_idps

  # Force terraform to wait for the IdP resource to exist before adding
  # its name to supported_identity_providers — otherwise the apply
  # transiently fails with "InvalidParameterException: Identity provider
  # not configured" because the client update races the IdP creation.
  depends_on = [aws_cognito_identity_provider.entra]

  explicit_auth_flows = [
    "ALLOW_USER_SRP_AUTH",
    "ALLOW_REFRESH_TOKEN_AUTH",
  ]

  access_token_validity  = 60
  id_token_validity      = 60
  # Lowered from 30 days to 1 day so a user disabled in Entra (or
  # removed from the M365 tenant) loses Praxis access within ~24h
  # rather than holding a valid refresh token for a month. Cognito
  # silently renews the access token in the background until the
  # refresh token expires; once it does, the SPA bounces the user
  # back through Microsoft, which fails for disabled users. Daily
  # re-auth is invisible if Entra is fine, immediate gate when it
  # isn't. See SECURITY_REVIEW finding M-2 (post-Phase-7 review).
  refresh_token_validity = 1

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
