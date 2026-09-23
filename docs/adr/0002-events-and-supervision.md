# 0002. Events and supervision at the core

Status: accepted.

## Context

SYNTHIA will run many things at once that must react to each other: audio
capture, a wake word detector, speech recognition, a reply being streamed and
spoken, a camera, background memory work. Speech from the user has to cancel a
reply in progress; a tracer has to see everything; a failing camera must not stop
the voice.

## Decision

**A typed publish-subscribe event bus.** Components publish frozen dataclass
events and subscribe by type, receiving subclasses too. Each subscription owns a
bounded queue and a consumer task: per-subscriber ordering, isolation between
subscribers, and an explicit overflow policy, backpressure or drop-oldest, chosen
by the subscriber.

**A one-for-one supervisor modelled on Erlang/OTP.** Each long-running part is a
service with `run(stop)`. Restart modes are permanent, transient and temporary.
Restarts are bounded by an intensity, a maximum number within a window, past
which the supervisor stops everything and escalates. The delay before a restart
grows exponentially with recent restarts.

## Options considered

- **Direct calls and callbacks.** Simplest at first, but every new observer
  means editing the producer, and cancellation has to be threaded through every
  call chain.
- **An external broker (Redis, NATS).** Right across processes or machines, but
  here every component lives in one process; a broker would add a service to
  run and a network hop to every event for no gain.
- **Unbounded queues.** Never block, but a slow subscriber then grows memory
  without limit. A bounded queue makes the failure visible and the policy
  deliberate.
- **Restart forever.** Hides a structural fault behind a loop that burns CPU and
  floods the log. The intensity limit turns it into one clear error.

## Consequences

Components stay independent and testable in isolation. The cost is indirection:
following what happens after an event means following its subscribers, which the
correlation id on every event and every log line is there to make practical.
