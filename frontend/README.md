# frontend — Vite + React SPA for anna-chat

## Status

Scaffold only. SPA lands in Phase 3.

## Planned stack

- Vite 5 + React 18
- TypeScript strict mode
- `oidc-client-ts` for Cognito Hosted UI auth
- `fetch` with `ReadableStream` for Lambda response streaming (no extra library needed)
- Minimal CSS — either plain CSS Modules or Tailwind, TBD

## Planned layout

```
frontend/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── index.html
├── public/
│   └── anna_logo.png    Copied from ../assets/ at build time
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── auth/            Cognito OIDC client setup
    ├── api/             Typed fetch wrappers for chat + conversations
    ├── components/
    │   ├── Chat.tsx
    │   ├── Message.tsx
    │   ├── ConversationList.tsx
    │   └── Layout.tsx
    └── styles/
```

## Build output

Built assets go to `dist/` and are synced to the SPA S3 bucket by the deploy pipeline. CloudFront invalidation runs automatically on deploy.
