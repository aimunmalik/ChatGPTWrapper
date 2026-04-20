# Deploy runbook

> **Status:** skeleton. Filled as each phase lands.

## Before the first deploy (one-time)

1. **Sign the AWS BAA**
   - AWS Console → AWS Artifact → Agreements → Search "BAA" → Sign for the account.
   - Save the signed PDF in ANNA's compliance records.

2. **Enable Bedrock model access**
   - AWS Console → Bedrock (us-east-1) → Model access → Request access.
   - Request: `Anthropic Claude Sonnet 4.6`, `Anthropic Claude Opus 4.7`.
   - Access is typically granted within minutes.

3. **Create an IAM Identity Center user for yourself**
   - Enable MFA (TOTP).
   - Grant `AdministratorAccess` on the account.
   - Do **not** create IAM users with long-lived access keys.

4. **Install local tools**
   - `terraform` ≥ 1.7
   - `aws` CLI v2
   - `node` ≥ 20
   - `python` 3.12
   - `gh` (GitHub CLI)

5. **Configure AWS CLI with Identity Center**
   ```
   aws configure sso
   ```
   Use the profile name `anna-chat`.

## Phase 1: Bootstrap and IaC foundation

*Details filled when Phase 1 lands.*

- [ ] Bootstrap Terraform state backend (one-time)
- [ ] Apply `infra/envs/dev` — creates VPC, KMS, Cognito, DynamoDB
- [ ] Verify: Cognito user pool exists, DynamoDB tables created, no security-hub findings

## Phase 2: Backend deploy

*Details filled when Phase 2 lands.*

## Phase 3: Frontend deploy

*Details filled when Phase 3 lands.*

## Phase 4: CI/CD

*Details filled when Phase 4 lands.*

## Rollback

*Per-phase rollback procedures filled as each phase lands.*
