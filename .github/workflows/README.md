# GitHub Actions workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | PR to `main`, push to `main` | Lint + pytest (backend), type-check + vite build (frontend), terraform plan (dev). No AWS writes. |
| `deploy.yml` | Push to `main`, manual dispatch | Builds Lambda zip, `terraform apply` on dev, builds SPA with production env vars, uploads to S3, invalidates CloudFront. |

Both workflows authenticate to AWS via OIDC — there are **no long-lived access keys** stored in GitHub. The assumed role is `anna-chat-dev-github-actions` (see `infra/envs/dev/ci.tf`).

## Required GitHub repository variables

Set these under **Settings → Secrets and variables → Actions → Variables tab** (not secrets — none of these are sensitive):

| Variable | Value |
|---|---|
| `AWS_REGION` | `us-east-1` |
| `AWS_ROLE_ARN` | `arn:aws:iam::743835780802:role/anna-chat-dev-github-actions` |
| `TF_STATE_BUCKET` | `anna-chat-tfstate-743835780802` |
| `TF_LOCK_TABLE` | `anna-chat-tfstate-lock` |
| `COGNITO_DOMAIN` | `anna-chat-dev-anna-dev-42.auth.us-east-1.amazoncognito.com` |
| `COGNITO_DOMAIN_SUFFIX` | `anna-dev-42` |
| `COGNITO_USER_POOL_ID` | `us-east-1_uUYWQeEHh` (used only by CI for cache-busting; deploy reads from TF output) |
| `COGNITO_SPA_CLIENT_ID` | `31i3ilng2oiie73eu2nsj22nsm` (same note) |
| `API_ENDPOINT` | `https://7x2axrhwff.execute-api.us-east-1.amazonaws.com` (same note) |

`COGNITO_DOMAIN`, `COGNITO_USER_POOL_ID`, etc. are used by the CI `frontend-build` job only to let the Vite type-check pass. The `deploy.yml` workflow reads the authoritative values from Terraform outputs at deploy time, so rotating them locally stays in sync automatically.

## When things go wrong

- **401 / `AccessDenied` assuming role**: trust policy on the role (in `infra/envs/dev/ci.tf`) requires the sub claim match `repo:aimunmalik/ChatGPTWrapper:ref:refs/heads/main` or `repo:aimunmalik/ChatGPTWrapper:pull_request`. If your org/repo ever renames, update `var.github_org` / `var.github_repo` in `terraform.tfvars`.
- **Terraform state lock held**: a previous run may have died mid-apply. Release via `aws dynamodb delete-item --table-name anna-chat-tfstate-lock ...` OR wait for the stale lock to TTL out.
- **Frontend build fails on a missing `VITE_*` env**: CI uses GitHub variables; deploy uses Terraform outputs. If a new env var is added to `frontend/src/config.ts`, update both workflow files + the GitHub variables list above.
