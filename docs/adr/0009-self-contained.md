# 0009. Self-contained: local first, outside services on command

Status: accepted. Supersedes the routing rule of
[0005](0005-model-gateway.md), which sent every request remote first.

## Context

SYNTHIA is meant to be a personal assistant that keeps working on its own
machine whatever happens elsewhere. Open-source code and model files that run
on the machine (the Python libraries, llama.cpp, the Qwen3.5-4B weights) are
part of it once installed. An outside service is different: OpenRouter, a
website or a hosted tool can be down, rate limited, changed or withdrawn, and
whatever is sent to it leaves the machine.

Decision 0004 put inference on the machine by default, but the code did not
follow it. The gateway refused to start without an OpenRouter key, even with
the local model installed. The router sent every request to the remote model
first and used the local one as a fallback, so every conversation left the
machine, and an OpenRouter outage changed what the chat did (fallbacks, an
open circuit, the budget). The chat also sent the key to OpenRouter at every
start, to look up the day's request cap.

## Decision

- **A request goes to the remote model only when it asks.** Each request
  carries `use_remote`, false by default. The router sends every other request
  to the local model.
- **A request that may not leave is never sent out.** If the local model is
  still loading, the request waits for it, up to the server's 180-second start
  timeout. If no local model can serve it (none is installed, or it cannot take
  images or tools), the request fails with that reason and how to ask for the
  remote model.
- **The permission travels on the request**, like the thinking level, so the
  chat, and later agents and voice, inherit it without any code of their own.
- **The remote is optional.** Without a key, SYNTHIA runs locally and the
  gateway has no remote at all. It refuses to start only when there is neither
  a key nor a local model.
- **Asked-for remote keeps the earlier rules**: remote first, local for
  background jobs, at the 10-request reserve, while the circuit is open or the
  rate limit would wait over 5 seconds, and one fallback before the first
  chunk.
- **Nothing else goes out before the command either.** The chat asks
  OpenRouter for the key's daily cap only after the first `/remote on`.
- In the chat, `/remote on` and `/remote off` switch the session, and
  `synthia chat --remote` starts with it on. `synthia doctor` reports a
  missing local model as a failure, since SYNTHIA cannot answer offline
  without one, and a missing key as fine.

## Options considered

- **Remote first, local as the fallback** (the previous behaviour). The
  strongest answers while the service is up, but every turn leaves the
  machine, and the service's state shapes every answer.
- **Local first, escalating to the remote when local cannot serve.**
  Convenient for images on a model without vision, but it sends a
  conversation out without being asked, and those requests still depend on
  the service.
- **A setting that turns remote on for every chat.** A standing yes in a file
  turns "on command" into "always", which is the first option again.

## Consequences

SYNTHIA works with no network and no key once its local model is installed.
On a laptop with an integrated GPU, with the server held to 2 threads while
other work ran, a first real turn after a cold start waited for the loading
server and answered locally in 38.2 seconds. Answers
are as good as the local model unless the person asks for more. Every later
capability that reaches outside, such as web access, hosted tools or other
agents, follows the same rule: local by default, outside only on command, and
its absence or failure never stops SYNTHIA working.
