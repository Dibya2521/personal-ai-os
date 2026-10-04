# 0011. The daemon: one process holds SYNTHIA, every interface is a client

Status: accepted.

## Context

Until 0.2.0, each `synthia chat` was SYNTHIA: it built the gateway, started
the local model's server and the MCP servers, held the conversation, and
stopped all of it when the chat ended. Three facts made that the wrong shape:

- Loading the local model takes 12 to 15 seconds on this laptop, longer after
  a restart, and the cost was paid by every chat.
- The interfaces still to come (a web page, voice) would each need the same
  start-up, and would each hold a separate copy of the model and the budget.
- The remote's daily budget, rate limit and circuit breaker only mean
  something when one process counts them.

The ROADMAP's Phase 3 goal is that SYNTHIA is always there, with the CLI as a
client.

## Options

1. **Keep SYNTHIA in each interface.** Nothing to build; every cost above
   stays, and two interfaces at once each load a model.
2. **A daemon with its own typed messages** over a socket. Full control, and
   everything a standard already solves (requests, replies, errors, cancel,
   requests from the server back to the client) built again.
3. **A daemon speaking JSON-RPC 2.0 over a WebSocket**, the protocol of LSP
   and MCP. Server-to-client requests carry tool approvals, notifications
   carry streamed text, and cancellation is defined. SYNTHIA already had a
   tested JSON-RPC implementation for MCP.
4. **HTTP endpoints plus server-sent events.** Easy to call from a browser;
   approvals would need a second channel back from the client.

## Decision

Option 3, served by FastAPI and uvicorn.

- **What runs where.** `synthia serve` holds the gateway, the local model's
  server (supervised, restarted when it dies), the MCP servers and the
  personas, started once. Each client connection is a conversation of its
  own: its history, persona and thinking level. The local model's server
  (four request slots in every measured run, llama.cpp's automatic default)
  and the remote budget are shared.
- **Finding it.** The daemon listens on a free port on 127.0.0.1, then writes
  `SYNTHIA_HOME/daemon.json` (port, process id, start time, version, a random
  token), readable only by its owner. A client trusts the file only if that
  process is still the one that wrote it (its start time is not later than
  the file's) and `GET /health` answers.
- **Who may talk to it.** `/health` answers anyone and says only that the
  daemon is up. The WebSocket at `/ws` needs the token as a Bearer header,
  compared in constant time, and any request with an `Origin` header is
  refused, so a web page in a browser cannot reach it even on the same machine.
- **Approvals stay with the person who asked.** A tool call that needs a yes
  is a request to the client whose turn it is. A client that has gone away is
  a no, so nothing that changes things or leaves the machine runs unattended
  (ADR 0010 is unchanged).
- **Lifetime.** `synthia chat` starts the daemon in the background when none
  is running and waits up to 90 seconds for it. It then runs until
  `synthia stop`, logout or shutdown. Stopping cancels running turns at once,
  stops the MCP servers and the local model, and removes `daemon.json`. A
  daemon of another version is restarted by the next chat if no other
  conversation is open in it; otherwise the chat says how.
- **What it remembers.** Conversations live in the daemon's memory until
  Phase 4 designs memory; a stopped daemon forgets them. Traces stay on disk:
  one per conversation, and one for the daemon itself, which records each
  start, end and restart of the local model's server.

## Consequences

- The second chat of the day starts without loading the model, and two
  clients at once share one model and one budget.
- While idle the daemon keeps the local model resident until `synthia stop`:
  measured on this laptop, the model's server held 5,502 MB and the daemon
  itself 75 MB.
- Killing the model's server under the daemon was measured too: it was
  started again half a second later, answered again about 17 seconds after
  that (loading at two threads on a busy laptop), and the daemon's trace
  holds the exit, its error and the second start.
- Anything on the machine running as the same user can read `daemon.json`
  and drive SYNTHIA; that is the same trust as reading its `.env`. Other users
  and web pages cannot.
- A browser interface cannot set a Bearer header on a WebSocket; Phase 11
  decides its authentication when the web UI exists.
- A restart of the daemon loses open conversations until Phase 4.
