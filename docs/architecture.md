# Architecture

SYNTHIA is organised as an operating system for a language model. This document
describes the whole target design and the order it is built in; each part is
refined when its phase begins, and this file changes with it.

## Principles

1. **Ports and adapters.** The core depends on protocols: a chat model, a
   speech recogniser, a detector, a vector index, a tool. Providers, models and
   operating systems are adapters behind them, so replacing one never reaches
   the core.
2. **Everything is an event.** Subsystems publish typed events on the kernel bus
   and subscribe to what they need. That is what makes interruption, tracing,
   replay and proactive behaviour possible without wiring callbacks through
   every layer.
3. **Supervised services.** Every long-running part runs as a service under a
   supervisor that restarts it, so one failing peripheral never stops the rest.
4. **Capabilities, not trust.** A tool declares what it needs (read files, fetch
   from the network, send input to the desktop). Grants are explicit, and a
   destructive capability needs a person's approval for each call. Anything a
   tool returns is data, never instructions.
5. **Budget is a resource.** Remote model calls, disk space and latency are
   metered the way an operating system meters CPU time. A call that would exceed
   the day's budget is refused before it starts, not failed half way.
6. **Optional weight.** The base install is small. Voice, vision and training
   are optional extras, and models download on first use into the data
   directory under a disk budget.
7. **Local first.** Whatever can run on the machine does. A remote model is used
   when a task needs it, and the routing decision is a rule first and a trained
   model later.

## Layers

```text
 interfaces   command line  ->  daemon API (HTTP, WebSocket)  ->  web interface  ->  tray app
     |
 agent        agent loop . planner . tool registry . capabilities . MCP client and server
     |
 cognition    model gateway: routing . budget . rate limit . retries . cache . structured output
 memory       context window . episodic store . vector and lexical indexes . knowledge graph
 knowledge    ingestion . chunking . retrieval . reranking . citations . evaluation
 perception   voice: audio . voice activity . wake word . speech to text . text to speech
              vision: camera and screen capture . change gating . detection . OCR . VLM
 action       operating system adapters . browser . computer use . scheduler
 learning     datasets . training . evaluation . export and quantisation . model registry
     |
 kernel       settings . logging . event bus . supervisor . errors
```

Dependencies point down only. The kernel imports nothing else from the project.

The cognition layer is built: [`gateway.md`](gateway.md) describes how a request
is routed, guarded, metered and answered, locally or remotely.

## Concurrency

One asyncio event loop owns all state. CPU-heavy inference (speech recognition,
detection, embeddings) runs in a bounded worker pool and returns its results as
events, so the loop never blocks. Each piece of shared state has exactly one
owning task; everything else reaches it by message.

## A spoken turn, end to end

```text
microphone -> audio frames (20 ms) -> ring buffer
  -> wake word -> voice activity starts (if SYNTHIA is speaking: interrupt her)
  -> streaming speech to text -> final transcript at end of speech
  -> agent turn -> router picks a model -> gateway streams tokens
  -> sentences -> text to speech -> speaker
every step carries the turn's correlation id and is timed in the trace
```

## Build order

| Phase | Release | Theme |
| --- | --- | --- |
| 0 | 0.1 | foundation: repository, gates, CI, kernel, command line |
| 1 | 0.2 | cognition: model gateway, budget, streaming chat |
| 2 | 0.3 | agency: agent loop, tools, capabilities, MCP |
| 3 | 0.4 | daemon: always-on service, local API |
| 4 | 0.5 | memory: context window, episodic store, own vector index |
| 5 | 0.6 | knowledge: retrieval with citations, evaluation |
| 6 | 0.7 | voice: full-duplex speech with interruption |
| 7 | 0.8 | vision: camera and screen perception |
| 8 | 0.9 | action: operating the computer with approval |
| 9 | 0.10 | learning: models trained here, and the pipeline behind them |
| 10 | 0.11 | proactive: routines, anomalies, initiative |
| 11 | 0.12 | interfaces: web and tray |
| 12 | 1.0 | hardening |

The order follows dependency: nothing is built before the layer it stands on
can carry it, and perception comes after cognition because a peripheral is only
as useful as the intelligence behind it.
