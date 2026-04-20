# Deploy runbook

Follow these steps in order. Most steps are one-time per environment.

## Before the first deploy (one-time)

### 1. Sign the AWS BAA

- AWS Console → AWS Artifact → Agreements → Search "BAA" → Sign for the account.
- Save the signed PDF in ANNA's compliance records.

### 2. Enable Bedrock model access

- AWS Console → Bedrock (us-east-1) → Model access → Request access.
- Request: `Anthropic Claude Sonnet 4.6`, `Anthropic Claude Opus 4.7`.
- Access is typically granted within minutes.

### 3. Enable IAM Identity Center + create your admin user

1. AWS Console → IAM Identity Center → Enable.
2. Choose AWS Organizations or stand-alone (stand-alone is fine for one account).
3. Create a user for yourself with your email address.
4. Create a permission set `anna-chat-admin` with the `AdministratorAccess` policy.
5. Assign your user to the account with that permission set.
6. Check the invitation email; set a password; enable MFA (TOTP via authenticator app).
7. **Do not create IAM users with long-lived access keys. Ever.**

### 4. Install local tools

| Tool | Version | Install |
|---|---|---|
| Terraform | ≥ 1.7 | https://developer.hashicorp.com/terraform/install |
| AWS CLI v2 | latest | https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html |
| Git | any modern | already installed |
| Node.js | ≥ 20 | https://nodejs.org/ (needed from Phase 3) |
| Python | 3.12 | https://www.python.org/downloads/ (needed from Phase 2) |
| GitHub CLI (`gh`) | any modern | https://cli.github.com/ |

### 5. Configure AWS CLI with Identity Center SSO

```bash
aws configure sso
```

Answer the prompts:
- SSO session name: `anna-chat`
- SSO start URL: (from your IAM Identity Center portal)
- SSO region: `us-east-1`
- Account: the 12-digit ANNA AWS account
- Role: `anna-chat-admin` (created in step 3)
- CLI default region: `us-east-1`
- CLI default output: `json`
- CLI profile name: `anna-chat`

Then authenticate:

```bash
aws sso login --profile anna-chat
export AWS_PROFILE=anna-chat   # or set it for each command
aws sts get-caller-identity    # should print your account + user
```

## Phase 1 — IaC foundation (first code-driven deploy)

Everything in this phase runs from your laptop. No CI yet — CI lands in Phase 4.

### Phase 1.1 — Bootstrap (one-time per account)

Creates the Terraform state backend (S3 bucket + DynamoDB lock table) and the GitHub Actions OIDC provider.

```bash
cd infra/bootstrap
terraform init
terraform plan -out=plan.tfplan
# Review the plan. Expect: 1 S3 bucket, 1 DynamoDB table, 1 IAM OIDC provider,
# plus policies and config resources (~10 resources total).
terraform apply plan.tfplan
```

Record the outputs — you'll paste them into the next step:

```bash
terraform output
```

You should see:
- `tf_state_bucket` — e.g. `anna-chat-tfstate-743835780802`
- `tf_lock_table`   — `anna-chat-tfstate-lock`
- `github_oidc_arn` — `arn:aws:iam::743835780802:oidc-provider/token.actions.githubusercontent.com`
- `account_id`      — `743835780802`

After this, **leave `infra/bootstrap/` alone**. Don't re-run it. The local `terraform.tfstate` is gitignored — back it up somewhere safe (password manager attachment, encrypted drive) so you can recover it if your laptop dies.

### Phase 1.2 — Dev environment

Creates KMS keys, VPC, Cognito user pool, DynamoDB tables.

**Prepare two local files** (both gitignored, so they stay on your laptop):

```bash
cd infra/envs/dev
cp backend.hcl.example backend.hcl
cp terraform.tfvars.example terraform.tfvars
```

Edit `backend.hcl` — the values should already match the outputs from Phase 1.1; confirm the bucket name has the right account ID.

Edit `terraform.tfvars` — change `cognito_domain_suffix` from `REPLACE_ME` to something short and random-ish, e.g. `anna-dev-k7q`. This shows up in the Cognito hosted-UI URL as `https://anna-chat-dev-anna-dev-k7q.auth.us-east-1.amazoncognito.com` and must be globally unique in the region.

Init, plan, apply:

```bash
terraform init -backend-config=backend.hcl
terraform plan -out=plan.tfplan
# Review carefully. Expect around 30-40 resources:
#   4 KMS keys (+ 4 aliases)
#   1 VPC, 2 private subnets, 1 route table, 2 associations
#   1 security group, 1 ingress rule
#   2 gateway VPC endpoints (S3, DynamoDB)
#   7 interface VPC endpoints (bedrock-runtime, kms, secretsmanager, logs, sts, ssm, monitoring)
#   1 Cognito user pool, 1 domain, 1 app client, 2 user groups
#   2 DynamoDB tables
terraform apply plan.tfplan
```

### Phase 1.3 — Create your first user

Dev is admin-create-only, so no self-signup. Create yourself as a user:

```bash
# Replace with your email and a strong temp password
aws cognito-idp admin-create-user \
  --user-pool-id "$(terraform -chdir=infra/envs/dev output -raw cognito_user_pool_id)" \
  --username "you@annaautismcare.com" \
  --user-attributes Name=email,Value="you@annaautismcare.com" Name=name,Value="Your Name" Name=email_verified,Value=true \
  --desired-delivery-mediums EMAIL
```

Check your inbox for the invitation. On first sign-in, you'll be forced to change the password and enroll TOTP.

> The Cognito Hosted UI isn't useful yet — there's no SPA to redirect back to. We'll exercise the full login flow in Phase 3.

### Phase 1.4 — Verify

```bash
terraform output
```

Sanity checks from the AWS Console:
- **KMS** → Customer managed keys → expect 4 keys named `anna-chat-dev-*`
- **VPC** → expect `anna-chat-dev-vpc`, 2 private subnets, 0 internet gateways, 0 NAT gateways
- **Cognito** → User pools → expect `anna-chat-dev-users` with MFA required
- **DynamoDB** → Tables → expect `anna-chat-dev-conversations` and `anna-chat-dev-messages`, both with PITR on and CMK encryption

Check Security Hub (if enabled): should see HIPAA conformance pack starting to collect findings. Resolve any `CRITICAL` before moving on.

## Phase 2 — Backend Lambda handlers

Builds and deploys two Lambdas behind a shared API Gateway HTTP API with a Cognito JWT authorizer.

### Phase 2.1 — Build the Lambda zip

```powershell
cd backend
python -m pip install --user -r requirements-dev.txt
python -m pytest -q              # 16 tests should pass
python build.py                  # produces backend/lambda.zip
```

The build script cross-compiles for Linux x86_64 (the Lambda target architecture). It works from Windows, macOS, or Linux.

### Phase 2.2 — Deploy

```powershell
cd ..\infra\envs\dev
terraform plan -out plan.tfplan
# Expect ~32 new resources: 2 Lambdas + roles + SGs + log groups,
# API Gateway + authorizer + stage + 4 routes with integrations and permissions.
terraform apply plan.tfplan
```

Record the new outputs:

```powershell
terraform output
```

Two matter most:
- `api_endpoint` — the base URL the frontend will call (e.g. `https://abc123.execute-api.us-east-1.amazonaws.com`)
- `lambda_functions` — the two function names

### Phase 2.3 — Smoke test

Unauthenticated call should return 401:

```powershell
curl -X POST https://<api-endpoint>/chat -H "Content-Type: application/json" -d '{}'
```

For a direct Lambda smoke test that bypasses API Gateway (uses a synthetic event with pre-validated claims), see the `Smoke tests` section of [../backend/README.md](../backend/README.md).

### Phase 2.4 — Troubleshooting

If Lambda invocations time out at 60s:
- Check CloudWatch log group `/aws/lambda/anna-chat-dev-chat` — the Lambda usually logs what it was doing.
- Verify the Lambda security group allows **outbound 443 to 0.0.0.0/0**. Gateway VPC endpoints (DynamoDB, S3) accept public IPs on SG rules but route traffic through the endpoint via the route table. An SG that only allows the VPC CIDR blocks gateway-endpoint traffic.

If `AccessDeniedException` on a Bedrock invoke:
- Confirm Bedrock model access is granted for the model ID configured in Terraform (`var.bedrock_model_id`, default `us.anthropic.claude-sonnet-4-6`).
- Bedrock console → Model access should show `Access granted` for Claude Sonnet 4.6 in us-east-1.

If CORS errors from the browser in Phase 3:
- Add your CloudFront domain to `cors_allow_origins` in `terraform.tfvars` and re-apply.

## Phase 3 — Frontend

*Runbook section added when Phase 3 code lands.*

## Phase 4 — CI/CD

*Runbook section added when Phase 4 code lands.*

## Phase 5 — Validation and prod promotion

*Runbook section added when Phase 5 lands.*

---

## Rollback

### Phase 1

To tear down dev (destroys all data — fine because we haven't loaded any):

```bash
cd infra/envs/dev
terraform destroy
```

To tear down the bootstrap (only if you're abandoning the whole project):

```bash
cd infra/bootstrap
# First, empty the state bucket manually in the S3 console — Terraform won't
# delete a bucket that has state files in it.
# Then:
terraform destroy
```

`prevent_destroy = true` is set on the state bucket, lock table, and OIDC provider to stop accidental deletes. You'll have to remove those lifecycle blocks temporarily if you really mean it.
