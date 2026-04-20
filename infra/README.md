# infra — Terraform for anna-chat

All AWS resources for anna-chat are defined here as Terraform.

## Status

Scaffold only. Terraform modules land in Phase 1.

## Planned layout

```
infra/
├── bootstrap/        One-time: creates the Terraform state backend (S3 + DynamoDB lock)
├── modules/          Reusable modules (kms, cognito, lambda, dynamo, network, edge)
└── envs/
    ├── dev/          Dev environment root module
    └── prod/         Prod environment root module
```

## Why this layout

- **`bootstrap/` is separate** because you can't store Terraform state in an S3 bucket that Terraform itself is about to create. Chicken-and-egg. Bootstrap is run once per AWS account, with local state, and never touched again.
- **`modules/` holds the reusable pieces.** Each module is narrow: `kms` makes one KMS key, `cognito` makes one user pool, etc. Root modules compose them.
- **`envs/dev` and `envs/prod` are separate root modules** rather than Terraform workspaces. This makes it harder to accidentally apply dev changes to prod — every prod change is an explicit `cd envs/prod && terraform apply`.

## Running

Detailed commands land in [../docs/DEPLOY.md](../docs/DEPLOY.md) as each phase ships.
