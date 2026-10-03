# 0010. Tool calling and permissions: the model asks, code decides

Status: accepted.

## Context

SYNTHIA answers more than text: it reads files, runs programs, fetches pages,
hands tasks to other agents and uses tools from MCP servers. Every one of
those is a request the model makes. The model's text can be steered by
whatever it reads (a web page, a file, a tool result), so anything the model
says about what it may do is a claim, not a permission.

Three facts shaped the design:

- The local model makes well-formed tool calls without help. Qwen3.5-4B on
  llama.cpp answered 60 of 60 correctly: ten prompts against six tools (with
  no-argument, string, integer and enum parameters), each run three times with
  thinking off and three times with thinking low, including two calls in one
  answer. So the client does not constrain the model's output with a grammar;
  it checks each call.
- Text inside a tool result can carry chat-template control tokens. A result
  holding them was read by the server as a new system turn and changed both
  answers in a test, until they were defused.
- A tool's own description of itself (a read-only hint from an MCP server, a
  plugin's declared scope) comes from the party the permission is meant to
  limit.

## Options

1. **Trust the model**: run every call it makes. Simple; one injected
   instruction can delete files or send data out.
2. **Let tools declare their own risk** and grant on that. Better, but a
   tool, or a server describing its tools, can declare itself harmless.
3. **Code outside the model decides, from facts SYNTHIA controls**: where the
   tool's work happens and whether it changes anything, as set by the code
   that builds the tool or by the person's own configuration.

## Decision

Option 3.

- **Every tool has a reach and an effect.** Reach is LOCAL (runs on this
  machine) or OUTSIDE (reaches a service SYNTHIA does not run). Effect is READ
  (changes nothing) or CHANGE (changes files or state). They are set where the
  tool is built, never by the model.
- **The rule is short.** LOCAL and READ runs without asking. Anything else asks
  the person before each call, one approval per call, so "outside on command"
  needs no session switch that one yes could leave open. A tool named in the
  denied list never runs. A refusal goes back to the model as the call's
  result.
- **What is approved is shown.** The question shows the tool and its
  arguments; arguments too long for one line are shown whole, and characters a
  terminal would act on are shown escaped, so nothing is approved unseen.
- **A server's or tool's own hints never lower the rule.** MCP servers are
  OUTSIDE and CHANGE unless the person's `mcp.toml` says otherwise.
- **Every call is checked.** Arguments are validated against the tool's
  schema; a call that does not fit goes back to the model as an error naming
  the field, and the model can try again. A failing tool returns its error as
  the result; it never ends the turn.
- **Tool output is data.** Each result reaches the model inside a
  `tool_result` block marked untrusted, with the closing tag and the
  chat-template token start defused, and lines shaped like instructions are
  flagged on the call's line. Quoting lowers how often a model follows injected
  text; the rule above is what makes it safe.
- **Limits bound every turn.** At most 8 model steps, the last offered no
  tools so the model must answer; 60 seconds per call unless the tool's code
  sets its own limit (an outside coding agent gets 10 minutes); and for
  unattended runs a 300-second deadline. In the chat the person is the
  deadline: the answer streams in view and Ctrl+C stops it.

## Consequences

- An injected instruction can at most make the model ask: everything that
  changes something or leaves the machine waits for the person.
- Reading inside the folders the person named needs no approval, so routine
  lookups do not train the person to say yes without looking.
- A program run on request (Python, an outside agent) is bounded in time,
  output and environment, but it is not an operating-system sandbox: it runs as
  the person. That is why it always asks.
- Each new tool must state its reach and effect; a wrong statement is a bug in
  SYNTHIA's code, reviewed like any other, not something a model or a server
  can change at run time.
