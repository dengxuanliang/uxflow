# Security Policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, use **GitHub Security Advisories** ("Report a vulnerability" on the Security tab of this repo) so the maintainers are notified privately. Include:

- a description of the issue and its impact,
- steps to reproduce,
- affected versions / commits,
- any suggested fix.

We will acknowledge receipt within 5 business days and aim for a fix or mitigation within 30 days.

## Secrets

- `.env` is gitignored. **Never commit real API keys**, LiteLLM keys, or credentials.
- The tracked `.env.example` carries only placeholder values; if you find a real-looking key (`sk-…`) committed anywhere, treat it as a credential leak and report it as above.
- If you accidentally commit a secret, rotate it immediately — do not rely on `git rm` alone, since the blob remains in history.
