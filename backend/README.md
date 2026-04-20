# backend — Python Lambda handlers for anna-chat

## Status

Scaffold only. Handlers land in Phase 2.

## Planned handlers

| Handler | Trigger | Purpose |
|---|---|---|
| `chat_stream` | Lambda Function URL (response streaming) | Verify JWT, read conversation context, invoke Bedrock with streaming, persist turns |
| `conversations` | API Gateway HTTP API | List/get/delete conversations for the authenticated user |
| `cognito_post_confirm` | Cognito trigger | On first sign-in, create the user's row in DynamoDB |

## Planned layout

```
backend/
├── pyproject.toml        Dependencies (boto3, pydantic, python-jose for JWT)
├── src/
│   └── anna_chat/
│       ├── __init__.py
│       ├── handlers/
│       │   ├── chat_stream.py
│       │   ├── conversations.py
│       │   └── cognito_post_confirm.py
│       ├── bedrock_client.py      Thin wrapper around bedrock-runtime
│       ├── ddb.py                 DynamoDB repository
│       ├── auth.py                Cognito JWT verification
│       ├── logging_config.py      Structured logs, PHI redaction
│       └── settings.py            Config from SSM Parameter Store
└── tests/
    ├── test_no_phi_in_logs.py
    ├── test_auth.py
    └── test_ddb.py
```

## PHI handling rules (enforced by tests)

- Message content never reaches the logger.
- Errors bubbling up to Lambda's default error handler are sanitized first.
- No `print()` calls (`ruff` rule enforced in CI).
