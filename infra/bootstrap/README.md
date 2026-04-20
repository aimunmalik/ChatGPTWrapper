# infra/bootstrap

One-time bootstrap for the Terraform state backend and GitHub Actions OIDC provider. Run once per AWS account. **Do not** include this in automated CI.

## What it creates

- **S3 bucket** to store Terraform state files (versioned, encrypted, public access blocked).
- **DynamoDB table** that Terraform uses to lock state during `apply` (prevents two people applying at the same time).
- **GitHub Actions OIDC provider** so that CI jobs in `github.com/aimunmalik/ChatGPTWrapper` can assume IAM roles without long-term access keys. The actual IAM roles are created in each environment's root module, not here.

## Why this is separate from the rest

Terraform can't store its state in an S3 bucket that Terraform itself is about to create — chicken-and-egg. So the bootstrap uses **local state** (kept out of Git via `.gitignore`). You run it once, get back a state-bucket name and lock-table name, then every other `terraform apply` in the repo uses those.

## How to run

### 1. Prerequisites

- AWS CLI configured with IAM Identity Center SSO profile named `anna-chat` (see [DEPLOY.md](../../docs/DEPLOY.md))
- Terraform ≥ 1.7 installed
- You are an admin on the target AWS account

### 2. Apply

```bash
cd infra/bootstrap
terraform init
terraform plan -out=plan.tfplan
# Review the plan output with a human before the next step
terraform apply plan.tfplan
```

### 3. Record outputs

```bash
terraform output
```

You'll get:
- `tf_state_bucket` — goes into every env's `backend.tf`
- `tf_lock_table` — goes into every env's `backend.tf`
- `github_oidc_arn` — goes into IAM role trust policies in Phase 4

Copy these into `infra/envs/dev/backend.tf` (there's an `example` version already).

### 4. Do not re-run casually

After the first successful apply, this directory should be left alone. The local state file `terraform.tfstate` is what proves the bootstrap ran. Keep it — back it up if needed — but don't regenerate these resources.

## Local state

The bootstrap's own state lives at `infra/bootstrap/terraform.tfstate` on your disk. It is gitignored. If you lose it, you'll have to manually import the existing AWS resources back into a new state (rare, but documented in the root [DEPLOY.md](../../docs/DEPLOY.md) recovery section when written).
