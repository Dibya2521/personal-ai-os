# The daemon

`synthia serve` runs SYNTHIA as one long-lived process: the gateway, the local
model's server, the MCP servers, the personas, memory and the knowledge
index, started once. Every
interface is a client of it. `synthia chat` is the first; a web page and
voice come later. Why it is built this way is in
[ADR 0011](adr/0011-daemon.md); this page is how it works.

## Commands

| Command | What it does |
| --- | --- |
| `synthia chat` | Uses the running daemon, or starts one in the background and waits up to 90 s for it, then holds a conversation there. `--resume` takes up the last remembered conversation. |
| `synthia serve` | Runs the daemon in the foreground (what `chat` starts in the background). Ctrl+C stops it. |
| `synthia status` | Says whether the daemon runs: version, since when, process id, port, other conversations open, and the local model's state. |
| `synthia stop` | Stops it: running answers are cancelled, the local model and the MCP servers stop, `daemon.json` is removed. Waits up to 30 s. |
| `synthia trace` | Shows a conversation's trace, or the daemon's own (local model starts, ends and restarts, consolidations of the facts it holds, and passes of the knowledge index). |
| `synthia memory consolidate` | Asks the running daemon to merge the facts it holds that the local model calls one, and prints how many pairs it asked about and merged. Exits 2 when no daemon runs, when it learns no facts, or when the run failed. |
| `synthia knowledge add <folder>` | Adds a folder for SYNTHIA to read (a folder holding `.obsidian` is read as a vault), or changes how one added before is read. `--private` marks it private (search shows its passages as such), `--public` undoes it; each `--exclude <path>` leaves out a folder or file inside it, and replaces the earlier list. Nothing is read until the next pass. |
| `synthia knowledge remove <folder>` | Stops reading a folder and forgets everything read from it. |
| `synthia knowledge list` | Shows each folder read: its name, kind (`files` or `vault`), whether it is private, where it is, and what it leaves out. |
| `synthia knowledge index` | Reads every folder now: new and changed files are read, deleted ones forgotten. Prints the counts in one line and names any folder that could not be listed (what was read from it before is kept). Exits 2 when the pass failed; Ctrl+C stops it and keeps what was read so far. |
| `synthia knowledge search "<question>"` | Prints the passages that best answer it, each with where it is (`name/path:start-end`), its score, `(private)` when it comes from a private folder, and its first 8 lines, control characters shown escaped. `--limit` (1 to 50, default 6) and `--variant` (`words`, `meaning`, `fused`, `fused:<word weight>`, each optionally followed by `+hop1x<weight>` or `+hop2x<weight>` to follow the links of the best 5 passages one or two steps) change what is found. `--deep` first has the local model rewrite the question in the words files likely use and draft a passage that answers it, searches for those too, and prints a line with how many it wrote and the seconds spent: about 35 s on a laptop CPU, so it is never the default. |

Each `synthia knowledge` command goes through the daemon, starting it if it
is not running, so one process writes the index. Without an embedding model
there is no index, and each exits 2 saying to run `synthia models install`.
The daemon loads no reranker, so a `--variant` that names one is refused.
[`knowledge.md`](knowledge.md) describes what is read and how it is found.

The local model's state is `ready` (answering), `starting` (loading, or
waiting to be restarted), `stopped` (stopped, or given up after too many
restarts) or `none` (no local model installed).

## Files

All under `SYNTHIA_HOME`.

| File | Contents |
| --- | --- |
| `daemon.json` | `port`, `pid`, `version`, `started`, `token`. Written after the port listens, readable only by its owner, removed on stop. |
| `traces/*.jsonl` | One per conversation, created at its first turn, and one for the daemon, created at its first record. |
| `memory.db` | Every finished turn of every conversation unless it is private, and the facts learned about the person (SQLite; [`memory.md`](memory.md)). |
| `knowledge.db` | The folders added with `synthia knowledge add`, and every passage read from them with its vector (SQLite). Opened only when an embedding model is installed. |
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
| `turn` | `text`, `plan` (bool), `deep` (bool: first run one deep `search_knowledge` for `text`, as `/deep` does; a conversation with no knowledge index fails with `-32001`), `images` (file paths) | the report: `route`, `model`, `prompt_tokens`, `completion_tokens`, `seconds`, `reasoning`, `confined` (true once the conversation holds private text: it stays on this machine until `reset`), `sources` (the citations, `name/path:start-end`, of the passages of the person's files that `search_knowledge` showed in the turn, in the order first shown, each once; `synthia chat` lists them in a dim line under the answer) |
| `think` | `level`: `off`, `low`, `medium`, `high`, `auto`, or null to ask | a reply |
| `remote` | `on`: true, false, or null to ask | a reply; in a conversation that holds private text, the choice is kept but the reply says remote stays off, with the reason, until `reset` |
| `private` | `on`: true (stop remembering this conversation), false, or null to ask | a reply |
| `forget` | `{}` | a reply; the last turn leaves the conversation and memory |
| `facts` | `{}` | a reply listing the facts held about the person, by number |
| `forget_fact` | `fact`: a number as `facts` lists it | a reply |
| `forget_about` | `words`: every fact and turn holding all of them is forgotten | a reply saying how many |
| `resume` | `{}`, before the conversation's first turn | a reply; the last remembered conversation's turns come back and its persona answers |
| `persona` | `key`: a persona, or empty to list them | a reply |
| `adjust` | `values`: trait name to a level from 0 to 1 | a reply |
| `budget`, `model`, `tools`, `reset` | `{}` | a reply |
| `status` | `{}` | `sessions`, `local` (the local model's state) |
| `consolidate` | `{}` | after the run: `asked`, `merged`, `seconds`, `scheduled`, `stopped`, `error` (null, or why it failed) |
| `knowledge_add` | `root` (an absolute folder path), `private` (bool, or null to keep), `excludes` (paths inside the folder, or null to keep) | the source: `name`, `root`, `kind` (`files` or `vault`), `private`, `excludes` |
| `knowledge_remove` | `root` | `removed`: false when no source has that folder |
| `knowledge_sources` | `{}` | `sources`, each as `knowledge_add` returns it, in the order added |
| `knowledge_index` | `{}` | after the pass, which does not wait for turns: `added`, `changed`, `unchanged`, `removed`, `unreadable`, `held_back`, `passed_over`, `chunks`, `cut`, `embedded`, `unavailable` (sources whose folder could not be listed), `seconds`, and `error` null; or `error` alone, saying why the pass failed |
| `knowledge_search` | `query`, `limit` (1 to 50, default 6), `variant` (null for the default), `deep` (bool, default false) | `passages` best first, each `citation` (`name/path:start-end`), `title`, `header`, `body`, `score`, `private`; and `withheld`. Private folders are searched too: the client is the person's own terminal. With `deep` also `expansion` (`queries` and `passage`, what the local model wrote), `seconds` (`expand`, `search`) and `note`; a model reply that never parsed gives the plain search with `expansion` null and a `note` saying so, and no local model is `-32001`. A variant that does not parse, or names a reranker (the daemon loads none), is `-32602`. |
| `stop` | `{}` | `{}`, then the daemon stops |

A reply is `{"notes": [...], "errors": [...]}`: lines to show.

### Daemon to client, while a turn runs

| Notification | Params |
| --- | --- |
| `chunk` | `text`, `reasoning`, `tool_calls` (bool: the model is asking for a tool), `progress` (`[read, total]` prompt tokens while the local model reads, else null) |
| `tool` | `name`, `arguments`, `result`, `ok`, `seconds`, `flags` (what in the result reads as an instruction) |
| `plan` | `kind`: `planned` (`steps`, `revised`), `step` (`number`, `text`) or `answer` |

A tool call that needs a yes is a request from the daemon to the client whose
turn it is: `approve` with `tool`, `arguments` and `confined`, true when the
conversation holds private text that the arguments may carry; `synthia chat`
then says so on a line before the question. Only a result of `true` runs the
call; any other answer, an error, or a client that has gone away is a no.

To stop a turn, the client sends `notifications/cancelled` with the turn's
`requestId`; the daemon cancels it, and the turn is dropped from the history.

### Errors

| Code | Meaning |
| --- | --- |
| `-32601` | No such method. |
| `-32602` | The params do not fit; the message names the field. |
| `-32001` | The answer failed, or a deep search's local model could not write its rewrites; the message says why. |
| `-32002` | An image could not be sent; the message names the file. |
| `-32003` | `consolidate` on a daemon that learns no facts: it needs a local model and an embedding model. |
| `-32004` | A `knowledge_` method on a daemon with no knowledge index: it needs an embedding model. |
| `-32005` | A folder knowledge refuses: missing, not a folder, inside or holding another source, or an exclude outside it. The message says which. |

## An exchange

```text
-> {"jsonrpc": "2.0", "id": 1, "method": "hello", "params": {}}
<- {"jsonrpc": "2.0", "id": 1, "result": {"version": "0.3.0", "persona": "SYNTHIA", "sessions": 1, "warnings": []}}
-> {"jsonrpc": "2.0", "id": 2, "method": "turn", "params": {"text": "what time is it?"}}
<- {"jsonrpc": "2.0", "method": "chunk", "params": {"text": "", "reasoning": "", "tool_calls": true, "progress": null}}
<- {"jsonrpc": "2.0", "method": "tool", "params": {"name": "current_time", "arguments": "{}", "result": "...", "ok": true, "seconds": 0.0, "flags": []}}
<- {"jsonrpc": "2.0", "method": "chunk", "params": {"text": "It is ...", "reasoning": "", "tool_calls": false, "progress": null}}
<- {"jsonrpc": "2.0", "id": 2, "result": {"route": "local", "model": "...", "prompt_tokens": 412, "completion_tokens": 18, "seconds": 2.4, "reasoning": "low", "confined": false, "sources": []}}
```

## Restarts

The local model's server runs under a supervisor. When it dies (a crash, or a
kill), the supervisor starts it again after a wait of half a second that
doubles with each restart inside a minute, and gives up when a sixth restart
would fall inside that minute; the daemon keeps serving the remote model
either way. Each start, end and restart is a record in the
daemon's trace, so `synthia trace` shows when the model went away, why, and
when it came back.

## Work beside the conversations

Besides serving, the daemon embeds the turns remembered without a vector
when it starts, learns facts from each remembered turn one at a time, and
consolidates the facts it holds after 10 minutes without a turn, at most
once an hour. All of it runs on this machine in the background, and no turn
waits for it; [`memory.md`](memory.md) describes it.

By the same rule, 10 minutes without a turn and at most once an hour after
the last pass ended, it reads the folders added with `synthia knowledge add`
again, so the index follows files that changed. That pass stops before its
next file, or its next round of embedding, when a turn starts or another
`synthia knowledge` command needs to change the index, so neither waits for
it; what it read is kept, and the next 10 idle minutes go on from there
([`knowledge.md`](knowledge.md#keeping-in-step) describes a pass). A pass asked for with
`synthia knowledge index` runs through, since the person is waiting for it.
Every pass, asked or not, stopped, failed or cancelled, is a record in the
daemon's trace.
