# The daemon

`synthia serve` runs SYNTHIA as one long-lived process: the gateway, the local
model's server, the MCP servers and the personas, started once. Every
interface is a client of it. `synthia chat` is the first; a web page and
voice come later. Why it is built this way is in
[ADR 0011](adr/0011-daemon.md); this page is how it works.

## Commands

| Command | What it does |
| --- | --- |
| `synthia chat` | Uses the running daemon, or starts one in the background and waits up to 90 s for it, then holds a conversation there. |
| `synthia serve` | Runs the daemon in the foreground (what `chat` starts in the background). Ctrl+C stops it. |
| `synthia status` | Says whether the daemon runs: version, since when, process id, port, other conversations open, and the local model's state. |
| `synthia stop` | Stops it: running answers are cancelled, the local model and the MCP servers stop, `daemon.json` is removed. Waits up to 30 s. |
| `synthia trace` | Shows a conversation's trace, or the daemon's own (local model starts, ends and restarts). |

The local model's state is `ready` (answering), `starting` (loading, or
waiting to be restarted), `stopped` (stopped, or given up after too many
restarts) or `none` (no local model installed).

## Files

All under `SYNTHIA_HOME`.

| File | Contents |
| --- | --- |
| `daemon.json` | `port`, `pid`, `version`, `started`, `token`. Written after the port listens, readable only by its owner, removed on stop. |
| `traces/*.jsonl` | One per conversation, created at its first turn, and one for the daemon, created when the local model first starts. |
| `memory.db` | Every finished turn of every conversation, kept unless the conversation is private (SQLite). |
| `logs/synthia.log` | The daemon's log, shared with the CLI. |
| `logs/llama-server.log` | The local model's server output. |

A client trusts `daemon.json` only if the process with that `pid` started no
later than `started` (so the id was not reused by another program) and
`GET /health` answers. Otherwise there is no daemon, and `synthia chat`
starts one.

## Connecting

The daemon listens on `127.0.0.1` only, on a free port.

- `GET /health` answers anyone with `{"status": "up"}` and nothing more.
- `/ws` is a WebSocket for one conversation. It needs
  `Authorization: Bearer <token>`, and refuses any request with an `Origin`
  header, which every browser sends, so a web page cannot reach it.

Each WebSocket is a conversation of its own, with its own history, persona,
thinking level and remote switch. Closing the socket ends the conversation and
cancels its running turn.

## Messages

Messages are JSON-RPC 2.0, one per WebSocket text frame.

### Client to daemon

| Method | Params | Result |
| --- | --- | --- |
| `hello` | `{}` | `version`, `persona`, `sessions` (conversations open, this one included), `warnings` from start-up |
| `turn` | `text`, `plan` (bool), `images` (file paths) | the report: `route`, `model`, `prompt_tokens`, `completion_tokens`, `seconds`, `reasoning` |
| `think` | `level`: `off`, `low`, `medium`, `high`, `auto`, or null to ask | a reply |
| `remote` | `on`: true, false, or null to ask | a reply |
| `private` | `on`: true (stop remembering this conversation), false, or null to ask | a reply |
| `forget` | `{}` | a reply; the last turn leaves the conversation and memory |
| `persona` | `key`: a persona, or empty to list them | a reply |
| `adjust` | `values`: trait name to a level from 0 to 1 | a reply |
| `budget`, `model`, `tools`, `reset` | `{}` | a reply |
| `status` | `{}` | `sessions`, `local` (the local model's state) |
| `stop` | `{}` | `{}`, then the daemon stops |

A reply is `{"notes": [...], "errors": [...]}`: lines to show.

### Daemon to client, while a turn runs

| Notification | Params |
| --- | --- |
| `chunk` | `text`, `reasoning`, `tool_calls` (bool: the model is asking for a tool), `progress` (`[read, total]` prompt tokens while the local model reads, else null) |
| `tool` | `name`, `arguments`, `result`, `ok`, `seconds`, `flags` (what in the result reads as an instruction) |
| `plan` | `kind`: `planned` (`steps`, `revised`), `step` (`number`, `text`) or `answer` |

A tool call that needs a yes is a request from the daemon to the client whose
turn it is: `approve` with `tool` and `arguments`. Only a result of `true`
runs the call; any other answer, an error, or a client that has gone away is
a no.

To stop a turn, the client sends `notifications/cancelled` with the turn's
`requestId`; the daemon cancels it, and the turn is dropped from the history.

### Errors

| Code | Meaning |
| --- | --- |
| `-32601` | No such method. |
| `-32602` | The params do not fit; the message names the field. |
| `-32001` | The answer failed; the message says why. |
| `-32002` | An image could not be sent; the message names the file. |

## An exchange

```text
-> {"jsonrpc": "2.0", "id": 1, "method": "hello", "params": {}}
<- {"jsonrpc": "2.0", "id": 1, "result": {"version": "0.2.0", "persona": "SYNTHIA", "sessions": 1, "warnings": []}}
-> {"jsonrpc": "2.0", "id": 2, "method": "turn", "params": {"text": "what time is it?"}}
<- {"jsonrpc": "2.0", "method": "chunk", "params": {"text": "", "reasoning": "", "tool_calls": true, "progress": null}}
<- {"jsonrpc": "2.0", "method": "tool", "params": {"name": "current_time", "arguments": "{}", "result": "...", "ok": true, "seconds": 0.0, "flags": []}}
<- {"jsonrpc": "2.0", "method": "chunk", "params": {"text": "It is ...", "reasoning": "", "tool_calls": false, "progress": null}}
<- {"jsonrpc": "2.0", "id": 2, "result": {"route": "local", "model": "...", "prompt_tokens": 412, "completion_tokens": 18, "seconds": 2.4, "reasoning": "low"}}
```

## Restarts

The local model's server runs under a supervisor. When it dies (a crash, or a
kill), the supervisor starts it again after a wait of half a second that
doubles with each restart inside a minute, and gives up when a sixth restart
would fall inside that minute; the daemon keeps serving the remote model
either way. Each start, end and restart is a record in the
daemon's trace, so `synthia trace` shows when the model went away, why, and
when it came back.
