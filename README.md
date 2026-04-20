# anna-chat

HIPAA-compliant Claude chat application for ANNA Health, deployed on AWS.

> **Status:** early scaffold. See `Phases` below for what's built and what's next.
> **Previous app:** the old Streamlit ChatGPT wrapper that lived in this repo has been preserved on branch `legacy-streamlit` and tag `pre-anna-chat-rebuild`. Check it out with `git checkout legacy-streamlit` if you need to reference or recover anything.

## What this is

A web chat interface, backed by Anthropic's Claude models via Amazon Bedrock, designed so that protected health information (PHI) can safely flow through it. Users log in with multi-factor auth, talk to the model, and conversation history is stored encrypted in the same AWS account that holds ANNA's Business Associate Agreement with AWS.

## Stack

| Layer | Choice |
|---|---|
| Model | Amazon Bedrock → Claude Sonnet 4.6 (default), Opus 4.7 (selectable) |
| Backend | Python 3.12 AWS Lambda |
| Frontend | Vite + React SPA, served by CloudFront + S3 |
| Auth | Amazon Cognito (Hosted UI, TOTP MFA required, admin-only provisioning) |
| Storage | DynamoDB (CMK-encrypted, PITR, 90-day message TTL) |
| Infra as Code | Terraform |
| Region | `us-east-1` |
| CI/CD | GitHub Actions with AWS OIDC (no long-lived keys) |

## Repository layout

```
anna-chat/
├── infra/        Terraform for all AWS resources
├── backend/      Python Lambda handlers
├── frontend/     Vite + React SPA
├── docs/         Architecture, deploy runbook, ops runbook, HIPAA posture, IR
└── assets/       Brand assets (logo, etc.)
```

## Phases

| Phase | Status | What it covers |
|---|---|---|
| 0. Scaffold | ✅ done | Repo structure, docs, architecture spec |
| 1. IaC foundation | ⏳ next | Terraform: state backend, KMS, VPC, Cognito, DynamoDB |
| 2. Backend | ⏳ | Lambda handlers: chat streaming, conversations CRUD, Cognito triggers |
| 3. Frontend | ⏳ | Vite + React chat UI with Cognito auth and streaming |
| 4. CI/CD | ⏳ | GitHub Actions with AWS OIDC, plan-on-PR, apply-on-main |
| 5. Validation | ⏳ | Dev-env smoke tests, pen test checklist, promotion to prod |

## Prerequisites (AWS side — one-time)

Before any deploy runs, the following must be true in the target AWS account:

- [ ] **Business Associate Agreement with AWS** signed (AWS Artifact → Agreements → AWS BAA)
- [ ] **Bedrock model access** granted for `anthropic.claude-sonnet-4-6-*` and `anthropic.claude-opus-4-7-*` in `us-east-1` (Bedrock console → Model access)
- [ ] **Root account MFA** enabled and root credentials sealed
- [ ] At least one **IAM Identity Center user** with admin permissions (no long-term root or IAM-user access keys)

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — services, data flow, HIPAA design decisions
- [HIPAA posture](docs/HIPAA_NOTES.md) — what's in scope, encryption, logging rules, BAA tracking
- [Deploy runbook](docs/DEPLOY.md) — step-by-step deploy to dev and prod
- [Ops runbook](docs/OPS.md) — monitoring, alerting, on-call
- [Incident response](docs/INCIDENT_RESPONSE.md) — breach-of-PHI checklist

## Collaboration model

Infra changes are written as Terraform in this repo, reviewed by a human, and deployed by a human. Access keys are never shared with AI assistants or other tools. Deploys go through CI using short-lived OIDC-federated credentials.

## Legacy

The prior Streamlit app (OpenAI + Google Vision OCR + ABA treatment analysis) is preserved at:
- Branch: `legacy-streamlit`
- Tag: `pre-anna-chat-rebuild`
- Commit: `27898cc` (at time of rebuild)

That code is not HIPAA-compliant and should not be used with real PHI.
