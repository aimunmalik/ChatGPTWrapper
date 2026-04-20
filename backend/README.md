# backend — Python Lambda handlers for anna-chat

## What's here

Two Lambdas wired behind a shared API Gateway HTTP API:

| Handler | Route | Purpose |
|---|---|---|
| `anna_chat.handlers.chat.handler` | `POST /chat` | Invokes Bedrock, writes user + assistant turns to DynamoDB |
| `anna_chat.handlers.conversations.handler` | `GET /conversations`, `GET /conversations/{id}/messages`, `DELETE /conversations/{id}` | CRUD over a user's conversation history |

Both Lambdas run in the VPC with no internet egress. DynamoDB traffic goes through a gateway endpoint; Bedrock, KMS, CloudWatch, STS, Secrets Manager, and SSM go through interface endpoints. PHI never leaves AWS's network.

## Layout

```
backend/
├── requirements.txt        Runtime deps (empty — everything we need is in the Python 3.12 Lambda runtime)
├── requirements-dev.txt    Test + tooling
├── pyproject.toml          pytest + ruff config
├── build.py                Cross-platform packaging script (produces lambda.zip)
├── src/
│   └── anna_chat/
│       ├── settings.py         Config from env vars
│       ├── logging_config.py   Structured JSON logs with PHI field-list redaction
│       ├── http.py             HTTP helpers + authorizer-claims extraction + Decimal-safe JSON
│       ├── ddb.py              Repository: Conversation, Message, queries
│       ├── bedrock_client.py   Wrapper around bedrock-runtime invoke_model
│       └── handlers/
│           ├── chat.py              POST /chat
│           └── conversations.py     Conversation CRUD
└── tests/
    ├── test_logging_no_phi.py   PHI must not leak into log output
    ├── test_http.py             HTTP helpers + auth claims extraction
    └── test_ddb_models.py       Sort-key ordering, dataclass serialization
```

## Auth model

API Gateway HTTP API has a JWT authorizer configured against the Cognito user pool (see `infra/modules/api`). Requests without a valid `Authorization: Bearer <token>` header are rejected by API Gateway before they ever reach a Lambda. Lambdas read the already-validated claims from `event.requestContext.authorizer.jwt.claims`.

We intentionally **don't** re-verify JWTs in the Lambda itself — that would be redundant and would require bundling `pyjwt[crypto]` and its `cryptography` wheel (~3 MB). If we later add a Lambda Function URL for streaming, that endpoint will need its own JWT verification (it doesn't sit behind API Gateway).

## PHI handling

- Bedrock invocation logging is **off** in prod (would record raw prompts and completions).
- CloudWatch Logs get metadata only. The `JsonFormatter` in `logging_config.py` strips a fixed set of PHI field names (`content`, `messages`, `body`, `prompt`, `completion`, `text`, `assistantMessage`, `userMessage`, `message`) from log records. This is enforced by `tests/test_logging_no_phi.py`.
- Message bodies live only in DynamoDB (CMK-encrypted, PITR, 90-day TTL) and in in-flight HTTP responses.

## Local development

Install dev deps and run tests:

```bash
cd backend
python -m pip install --user -r requirements-dev.txt
python -m pytest -q
python -m ruff check src/ tests/
```

Build the deploy zip (produces `backend/lambda.zip`, which Terraform reads):

```bash
python build.py
```

The build script forces Linux x86_64 wheels via `pip install --platform manylinux2014_x86_64`, so you can build from Windows/macOS and still get a working Lambda package.

## Smoke tests

After deploy, you can invoke the chat Lambda with a synthetic API Gateway event:

```bash
cat > /tmp/smoke.json <<'JSON'
{
  "version": "2.0",
  "routeKey": "POST /chat",
  "rawPath": "/chat",
  "headers": {"authorization": "Bearer synthetic"},
  "requestContext": {
    "http": {"method": "POST", "path": "/chat"},
    "authorizer": {"jwt": {"claims": {"sub": "smoke-user", "email": "x@y.z"}}}
  },
  "body": "{\"message\":\"Say hi.\"}",
  "isBase64Encoded": false
}
JSON

aws lambda invoke --function-name anna-chat-dev-chat \
  --cli-binary-format raw-in-base64-out \
  --payload fileb:///tmp/smoke.json /tmp/out.json

cat /tmp/out.json
```
