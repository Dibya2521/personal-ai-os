# Memory

SYNTHIA remembers every finished turn of every chat, on this machine. It
finds earlier turns by their words and by their meaning, learns lasting facts
about the person from what they say, and keeps each request inside the
model's context window however long a chat runs. Why it is built this way is
in [ADR 0012](adr/0012-memory.md); this page is how it works, with each
number and the command that measures it.

## What each model adds

Memory works with no model at all, and gains a part with each one installed.

| Installed | What memory does |
| --- | --- |
| nothing | keeps every turn, finds turns by their words, drops the oldest turns that no longer fit the window |
| an embedding model | also finds turns by meaning, and puts close turns from other chats in front of each question |
| the local model | also folds the turns that leave the window into a running summary |
| both | also learns facts about the person, pins them in each chat's system message, and merges repeats |

`synthia models install` installs the local model and the default embedding
model, all-MiniLM-L6-v2 quantised to 8 bits (22.4 MB, vectors of 384
numbers). Facts and summaries are written by the local model only: their
requests never allow the remote model, whatever `/remote` says.

## What is kept

Everything is in one SQLite file, `SYNTHIA_HOME/memory.db`, in write-ahead
logging mode, so a reader never waits for a writer. Times are stored in UTC.

| Table | One row per | Holds |
| --- | --- | --- |
| `conversations` | chat | when it started and its persona; made at the chat's first remembered turn |
| `turns` | finished turn | the question and the answer whole, when it was asked, the persona, the route and the model, and the tool calls' names and arguments as one text cut to 2,000 characters, which is all the full-text index sees of them |
| `tool_calls` | tool call | its name, arguments and result, each cut to 2,000 characters, and whether it succeeded |
| `turns_text`, `facts_text` | turn, fact | full-text indexes (SQLite FTS5, Porter stemming), kept in step by triggers |
| `turn_vectors` | turn | the vector of the question's meaning, and the embedding model that made it |
| `facts` | fact | the sentence, `fact` or `preference`, the turn it came from, `since`, and `until` with `replaced_by` once a newer fact changed it |
| `fact_vectors` | fact | the vector of the sentence |
| `facts_pending` | turn | a turn waiting to be learned from, and how many tries failed |
| `facts_apart` | pair of facts | a pair the local model judged two different facts |

A stopped or failed turn is not kept, and nothing is kept while a chat is
private. Forgetting a turn takes its tool calls, its vector, the facts learned
from it and their vectors with it, in the same statement.

## A turn, through memory

1. **Pinned facts.** The facts held about the person, newest first, go into
   the system message under a header saying they were learned in earlier
   conversations and may have changed. Whole facts only, up to 1,000 tokens.
   A chat takes them once and keeps them, since a system message that changes
   makes the local server read the whole prompt again; it takes them afresh
   after `/reset`, and when a fact in them is forgotten, in this chat or
   another.
2. **A reminder.** With an embedding model, up to three turns from other
   chats that score at least 0.55 (the score is below) go in front of the
   question, under a header saying they come from SYNTHIA's memory and were
   not written by the person. The chat's history keeps the question alone,
   so reminders never pile up in the window.
3. **The window.** The request is counted (below), and the oldest turns
   leave, whole, until it fits with a quarter of the window kept free for
   the turn's tool results, thinking and answer. A turn leaves with its tool
   calls and their results, so the model never sees a call without its
   result.
4. **The summary.** With a local model, the turns that left are folded into
   a running summary in the background, at most 1,500 tokens, which the
   system message carries from then on. Without one they are dropped.
5. **After the answer.** The turn is written to the store, its question is
   embedded and kept for search, and, with facts being learned, it joins the
   queue of turns to learn from. A write that fails is logged and the chat
   goes on: memory is kept beside the answers, never in their way.

### Counting tokens

A request is counted in characters: its messages, its tool calls and its
tool definitions. Characters become tokens at a rate learned from the model.
Once a turn, the rate moves 30 % of the way towards the characters per token
of the turn's first request, from the prompt size the model reported for it,
and it stays between 2 and 6; a `/plan` turn does not move it. It starts at
2.5, below every figure measured in a real chat (2.67 to 2.91, with the tool
definitions and the template's markers counted), so for chats like that one
an estimate made before the model has reported, on a chat's first turn or a
resumed chat, errs towards too many tokens. Requests of another kind, mostly
code or another language, were not measured and may run below it. Each
message adds 8 tokens for the chat template's markers, and each image 1,500
(not measured: a vision model's cost per image depends on its size).

## Finding earlier turns

The model has a tool, `search_memory`, for "what did I tell you about my cat
last week?". It reads only this machine's memory, so it runs without asking.
It takes words and optional first and last days, in this machine's time zone,
the one the clock tool names: the model reads the clock and turns "last week"
into dates itself. It returns up to 8 turns, best first, each with its day,
time, conversation and turn number, the question and the answer cut to 300
characters each, and the names of the tools it used. With no words, it
returns the newest turns in the dates.

Each turn found by either its words or its meaning gets one score:

```text
cosine similarity of the questions + 0.1 * bm25 / (ln(2N + 1) * (k1 + 1))
```

The first part is meaning: the cosine of the query's vector and the turn
question's vector. The second is words: the turn's BM25 score from SQLite's
full-text index (k1 1.2) over its question, answer and tool calls, divided by
the most one query word can score among N turns, so a query whose only shared
words are "my" and "what" keeps the small word score it deserves. Words put
forward their best 50 turns and meaning its closest 50, and each is scored on
both parts. Without an embedding model, words alone decide.

Each turn's vector is of the person's question alone: what they said is what
"what did I tell you" asks about, and a long answer averaged in would make
the vector about the answer.

The vector search is exact: every vector is compared with the query in one
matrix product. On a laptop CPU it took 1.9 ms at 10,000 turns and 6.1 ms at
50,000, while the embedding model took 4.8 ms a text. SYNTHIA also has its
own HNSW index (`synthia.memory.hnsw`), built and measured against exact
search for the store sizes where exact search stops being cheap; the numbers
are below.

## Facts

### Learning

After each remembered turn, with both models installed, a background worker
in the daemon shows the local model the person's words, SYNTHIA's answer (cut
to 1,000 characters) and the five held facts closest to the question, and
asks for the lasting facts the exchange adds about the person: at most five,
one short sentence of at most 200 characters each, in the third person, each
a `fact` or a `preference`, each with the number of a shown fact it changes,
if any. Two worked examples in the request show it one exchange that states
facts and one that states none. llama.cpp holds the reply to that shape while
generating it; SYNTHIA checks it again, and asks at most twice more, showing
the model what was wrong, when it does not match.

- A new fact that changes a shown one ends the old one: it keeps its text,
  gets the day it stopped being true, and points to the fact that replaced
  it.
- A new fact whose vector is at least 0.90 similar to a held one is the same
  fact again and is not kept.
- The queue of turns to learn from is a table, so a stop or a crash loses
  none. A turn leaves it when it has been learned from, when the model's reply
  never matched the shape, or after three failures of another kind. While the
  local model is down, turns wait.
- Private turns are never learned from.

### Consolidation

The daemon also tidies the facts it holds. When no turn has come for 10
minutes, at most once an hour, and only when a turn was remembered since the
last run or that run failed, it:

1. Takes each pair of held facts at least 0.80 similar, most similar first,
   skipping pairs judged before, and asks the local model whether the two say
   the same thing, in both orders. Both yes: the older is ended as replaced by
   the newer. A no in either order: the pair is remembered as two facts and
   never asked about again. No valid answer: the pair is asked at the next
   run. Each fact takes part in one merge at most per run.
2. Merges the full-text indexes' segments (FTS5 `optimize`) and refreshes
   SQLite's query planner statistics (`PRAGMA optimize`), after a pass that
   ended without an error.

A run the daemon starts on its own gives way to the person: it stops before
its next question once a turn is waiting, and what it left is asked at the
next run. Each run is a `consolidation` record in the daemon's trace with
the pairs asked, the facts merged, the seconds taken, and the error if it
failed. `synthia memory consolidate` asks the running daemon for a run now,
waits for it to finish, and prints the same counts.

## Forgetting

| In the chat | What it does |
| --- | --- |
| `/private on` | stops remembering this chat from here on; `/private off` starts again |
| `/forget` | takes the last turn out of the chat and out of memory |
| `/facts` | lists the facts held, by number |
| `/forget fact <number>` | forgets one fact |
| `/forget about <words>` | forgets every fact and every turn holding all the words, and the facts learned from those turns, and says how many |

Forgetting goes by words, not meaning, so it is exact and can be explained:
`/forget about Ananya Pune` forgets what holds both words and nothing else.
A forgotten turn or fact leaves search and recall at once, and the pinned
facts of every open chat at its next turn. What was said in the current chat
stays in its history until `/reset`.

## Resuming

`synthia chat --resume` takes up the last conversation: the one holding the
newest remembered turn, answered by the persona of that turn if one of that
name still exists (`--persona` given too still wins). Its turns come back as
what the person asked and what SYNTHIA answered, without their tool calls,
which are kept cut short. Only the newest turns that fit the window come
back, and older ones are not summarised, since all of them at once could be
more than the model can read; the chat says how many stayed out, and
`search_memory` still finds them. New turns are remembered as part of the
resumed conversation.

## Commands

| Command | What it does |
| --- | --- |
| `synthia chat --resume` | takes up the last conversation, as above |
| `synthia memory consolidate` | asks the running daemon to consolidate the facts now; exits 2 when no daemon runs, when the daemon learns no facts (no local or no embedding model), or when the local model failed |
| `synthia trace` | shows the daemon's consolidation runs among its records; `--list` counts them |

## Numbers

Each was measured with the command beside it, on a laptop CPU while other
work kept it busy; the commands print the load with the numbers. Timings move
with the load and the machine; recalls and sizes do not.

**Own HNSW against exact search** (`uv run python benchmarks/hnsw_recall.py
--size 1000 --size 10000 --size 50000`): recall@10 is the share of the exact
ten nearest neighbours the index returns, over 200 queries of 384-number
vectors drawn in clusters, the shape sentence vectors take. The index links
each vector to up to 16 others on each of its layers (32 on the bottom one)
and weighs 100 candidates while building; a query weighs `ef_search`
candidates (64 by default).

| Vectors | recall@10 at ef_search 10 / 20 / 40 / 80 / 160 | HNSW query at 40 | Exact query | Build per vector |
| --- | --- | --- | --- | --- |
| 1,000 | 0.928 / 0.9885 / 1.000 / 1.000 / 1.000 | 1.86 ms | 0.14 ms | 6.3 ms |
| 10,000 | 0.915 / 0.9855 / 0.999 / 1.000 / 1.000 | 2.87 ms | 1.94 ms | 8.0 ms |
| 50,000 | 0.9235 / 0.9865 / 0.9995 / 1.000 / 1.000 | 5.20 ms | 6.13 ms | 14.2 ms |

Timings are from one run, with other work keeping the CPU 26 to 74 % busy,
and move from run to run; the recalls do not, since the vectors are drawn
from a fixed seed. Up to 50,000 turns, exact search costs at most about what
embedding a text does, and it is always right, so memory searches exactly.

**Embedding models** (`uv run python benchmarks/embedding_recall.py`): 40
questions asked in other words against 160 turns. all-MiniLM-L6-v2 (8-bit)
put the right turn first for 0.625 of them and in the first five for 0.925,
at 4.8 ms a text; bge-small-en-v1.5 0.625 and 0.875, at 6.2 ms; BM25 words
alone 0.075 and 0.2.

**Recall on a 60-question set** (`uv run python benchmarks/hybrid_recall.py`):
200 made-up turns, 40 questions asked in other words than what was said
("What's the name of my pet?" for "My cat is called Miso") and 20 naming a
rare word ("Who has the keys to kestrel-07?"), with 20 turns that share a
topic with a named one under another name. Recall@1 is how often the right
turn came first.

| Method | Other words: recall@1 | recall@5 | Rare word: recall@1 | recall@5 |
| --- | --- | --- | --- | --- |
| words | 0.025 | 0.125 | 0.70 | 1.00 |
| meaning | 0.50 | 0.925 | 0.90 | 1.00 |
| both (SYNTHIA's score) | 0.50 | 0.925 | 0.95 | 1.00 |

At the reminder's floor of 0.55, the right turn reached it for 25 % of
questions, and for 6.7 % some other turn would have, had the right one never
been said: the floor keeps reminders rare and mostly right, and leaves the
rest to `search_memory`.

**Disk per 1,000 turns** (`uv run python benchmarks/store_size.py`): turns
of English sentences cut to each length, with no tool calls or facts.

| Question / answer characters | Text alone | Words only | With vectors | After index upkeep (free inside) |
| --- | --- | --- | --- | --- |
| 100 / 500 | 0.57 MB | 1.30 MB | 3.25 MB | 3.68 MB (0.48 MB) |
| 200 / 1,500 | 1.62 MB | 3.38 MB | 5.31 MB | 6.48 MB (1.25 MB) |
| 400 / 4,000 | 4.20 MB | 7.47 MB | 9.40 MB | 12.18 MB (2.88 MB) |

A megabyte here is 1,048,576 bytes, as in `synthia models`.

With words only the file is about twice the text it holds, and a vector adds
about 2 KB a turn: two vectors of 1,536 bytes fill one 4,096-byte page.
Index upkeep writes each full-text index anew in one piece; the old one's
pages stay in the file as free pages, which later turns fill before the file
grows again.

**Characters per token** (`uv run python benchmarks/token_ratio.py --threads
2`): a six-turn chat on qwen3.5-4b, as the default persona with SYNTHIA's
own tools, thinking off, run twice. Every turn's figure was between 2.669
and 2.912 characters a token (the first turn, the same prompt both times,
measured 2.669 both times), and from its start of 2.5 the window's rate
ended at 2.785.

**Facts.** On 24 made-up exchanges with the expected facts written before any
run, the local qwen3.5-4b found facts with precision 0.952 and recall 0.95 (a
found fact matching an expected one at cosine 0.7) and handled 8 of 8 changed
facts (`benchmarks/fact_extraction.py`, October 2026). On 28 pairs that say
the same fact and 28 that do not, also written before any run, the merge
question merged 23 of the first and none of the second
(`benchmarks/fact_merging.py`, October 2026), at a median of 14.1 s per
question.
