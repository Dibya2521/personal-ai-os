# 0004. Local first, optional weight

Status: accepted.

## Context

SYNTHIA has to run on an ordinary laptop: no CUDA GPU, at most an integrated
GPU, and limited disk space for model files. The remote models
available at no cost are rate limited to 20 requests a minute and 50 a day, and
the free catalogue changes.

## Decision

- **Inference runs on the machine by default**: quantised language models, and
  ONNX or OpenVINO models for speech and vision on the CPU and integrated GPU.
- **Remote models sit behind one gateway**, reached through adapters, with a
  daily request budget and a per-minute rate limit taken from configuration.
  The gateway refuses a request that would exceed the budget before sending it.
- **Tests never call a remote model.** Recorded responses are replayed, so CI
  spends nothing and runs offline.
- **Heavy capabilities are optional extras**, and model files download on first
  use into the data directory, checked against a disk budget.

## Options considered

- **A paid frontier API for everything.** The strongest results soonest, but it
  costs money per request, sends every conversation off the machine, and leaves
  little engineering to show.
- **A rented GPU server.** The most capacity, at a running cost and with a
  permanent network dependency.

## Consequences

Model choice is a configuration change, not a code change, and a model leaving
the free catalogue affects one adapter. The budget and the local models together
decide what SYNTHIA can do in a day, so both are measured rather than assumed.
