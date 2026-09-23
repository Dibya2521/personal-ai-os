# 0003. Configuration and secrets

Status: accepted.

## Decision

- Configuration is read from `SYNTHIA_` environment variables, then a `.env`
  file, then defaults, into one frozen, typed settings class validated at
  start-up.
- `.env.example` is committed and documents every variable with an empty value.
  A test fails if it and the settings class disagree.
- Secrets are `SecretStr`: masked in a repr, and read only by an explicit call.
- A validation error names the variable and the problem, never the value.
- The log formatter masks every secret value in the fully formatted text, after
  arguments, extra fields and tracebacks have been rendered into it, and repeats
  until none is left, because the mask beside leftover text can rebuild a
  secret.
- Commits pass `detect-secrets`, and a guard that rejects staged files matching
  private patterns kept outside version control.

## Options considered

- **A committed configuration file.** Works for structure humans edit, but it
  invites secrets into history and needs a code or file change per machine.
- **Plain `os.environ` lookups.** No validation; a typo or a wrong type fails
  late, in the middle of a conversation.
- **Redacting the message template only.** Misses secrets passed as arguments,
  in extra fields or inside exception text, which is where they usually are.

## Consequences

A misconfiguration fails at start-up with a message that is safe to show. A
secret has to get past three independent layers to reach a log line or the
public history.
