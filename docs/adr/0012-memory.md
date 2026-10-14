# 0012. Memory: one SQLite file, words and meaning together, facts written on this machine

Status: accepted.

## Context

Before memory, SYNTHIA forgot a conversation when it ended, and a long one
grew until the model could not read it: every turn of the history went into
every request. The goal is that SYNTHIA remembers across conversations and
knows what to keep in the context window: "what did I tell you about X last
week" must be answered from what was said, and SYNTHIA's own vector index
must be measured against brute force before anything relies on it.

Facts that shaped the design:

- Everything runs on the person's machine first (ADR 0004, 0009), usually a
  laptop CPU that is also doing the person's other work. One daemon holds
  SYNTHIA (ADR 0011), so memory has one owner process and needs no server of
  its own.
- What a person asks about later is either a rare word (a host name, a
  ticket number, a person) or a meaning in other words ("what's my pet
  called?" for "my cat is called Miso"). On a 60-question set, word search
  put the right turn first for 70 % of the rare-word questions and 2.5 % of
  the others; a sentence embedding model put it first for 90 % and 50 %.
- What a person tells SYNTHIA about themselves is the most private thing it
  holds. Whether two statements contradict cannot be read from their vectors:
  "Ananya lives in Pune" and "Ananya lives in Mumbai" measured 0.859 similar
  with the default embedding model, closer than most rewordings of one fact.
- The local model is slow on a CPU: in October 2026, a fact-learning request
  took 7.1 to 29.6 s (median 14.5, 32 requests, two threads on a busy
  laptop; `benchmarks/fact_extraction.py`), and a yes or no question about two
  facts a median of 14.1 s (`benchmarks/fact_merging.py`). Anything it does
  for memory has to run in the background and never twice for the same
  answer.

## Options

**Where memory lives.**

1. **A vector database server** (Qdrant, Chroma, Milvus): vector search
   built in, but a second process to install, start and keep in step, and
   word search and transactions handled elsewhere.
2. **Postgres with pgvector**: both searches and transactions, and a database
   server on every machine SYNTHIA runs on.
3. **SQLite with FTS5**: one file, in Python's standard library; BM25 word
   search, transactions, triggers and cascading deletes; no vector search.

**How meaning is searched.**

1. **A library index** (FAISS, hnswlib): fast at any size, one more
   dependency, built for millions of vectors.
2. **SYNTHIA's own HNSW index**: built and measured against brute force,
   approximate (recall@10 0.999 at 10,000 vectors, weighing 40 candidates a
   query), and a second structure to keep in step with every forget.
3. **Exact search**: one matrix product over every vector, always right;
   1.9 ms at 10,000 turns and 6.1 ms at 50,000, against 4.8 ms for the
   embedding model to embed a text.

**How the two searches are combined.**

1. **Reciprocal rank fusion** (k 60), the usual choice: each list's first
   result counts the same however weak the list is. On 40 questions asked in
   other words, it cut how often the right turn came first from 0.500 to
   0.125 (`benchmarks/hybrid_recall.py`, October 2026).
2. **Meaning alone**: 0.90 for rare words against 0.95 combined.
3. **A score sum on a fixed scale**: cosine plus 0.1 times the BM25 score
   divided by the most one word can score in the store. In the same
   benchmark, any weight from 0.05 to 0.15 was never worse than meaning alone
   on either kind of question.

**What is learned about the person.**

1. **Nothing beyond the turns**: search finds what was said, but nothing
   carries a preference into the next chat unless it is searched for.
2. **Facts written by the remote model**: better writing, and the person's
   life sent to a provider for every turn.
3. **Facts written by the local model**, in a fixed JSON shape, shown the
   five held facts closest to the person's words so it can say which one a
   new fact changes.

**How repeats among facts are merged.**

1. **A lower similarity line alone**: on 12 pairs of different facts written
   for `benchmarks/fact_extraction.py`, a line of 0.85 merged 2, and a wrong
   merge loses a true fact.
2. **The local model asked yes or no** about each close pair, in both
   orders, its "no" remembered so no pair is asked twice.

## Decision

- **One SQLite file**, `memory.db`, in write-ahead logging mode, a
  connection per call, the work done off the event loop. Turns, tool calls,
  facts and their vectors are rows; full-text indexes follow by trigger;
  forgetting a turn removes everything that came from it by cascade.
- **Exact vector search** while stores are this size, with SYNTHIA's own
  HNSW index built, tested and measured for the size where it pays
  (recall@10 against brute force at `ef_search` 40: 1.000 at 1,000 vectors,
  0.999 at 10,000, 0.9995 at 50,000). Each turn's vector is of the person's
  question alone.
- **Score fusion on a fixed scale**, weight 0.1, over each side's best 50.
- **The default embedding model is all-MiniLM-L6-v2 quantised to 8 bits**: on
  40 questions asked in other words it put the right turn first as often as
  bge-small-en-v1.5 (0.625) and in the first five more often (0.925 against
  0.875), at 4.8 ms a text against 6.2, and 22.4 MB against 127.6 MB.
- **Facts are written, compared and merged by the local model only.** A new
  fact at least 0.90 similar to a held one is not kept; a fact that changes
  another ends it with a date and a pointer to its replacement, so history is
  kept. Pairs at least 0.80 similar are asked about when the daemon is idle,
  at most hourly; a pair is merged only when both orders say yes.
- **No decay.** Nothing measures whether a fact is used, and silently ending
  facts would take control from the person. The pinned facts are cut to 1,000
  tokens, newest first, so an old fact already leaves the prompt as new ones
  arrive.
- **The window is counted, not filled.** Characters are turned into tokens at
  a rate learned from each answer's reported prompt size, so every model's
  tokenizer is followed without being loaded. The rate starts at 2.5
  characters a token, below every turn of a real chat measured on the local
  model (2.67 to 2.91), so for such chats a count made before the first
  report errs high. A quarter of the window is kept free; old turns leave
  whole and are summarised by the local model, never the remote.
- **Forgetting goes by words, not meaning**, so it is exact and explainable.

## Consequences

- "What did I tell you about X last week" works: the model reads the clock,
  calls `search_memory` with dates, and gets the turns. A daemon restart no
  longer loses a conversation for good: `synthia chat --resume` takes it up.
- Memory costs disk: 5.3 MB per 1,000 turns of a 200-character question and
  a 1,500-character answer, vectors included. Index upkeep writes each
  full-text index anew without a VACUUM, which would rewrite the whole file,
  so the old index's pages stay in the file as free pages that later turns
  fill (1.25 MB per 1,000 such turns).
- Facts need both the local model and an embedding model. Without them,
  memory still keeps and finds turns. Without the local model, the turns
  that leave the window are dropped instead of summarised.
- Facts are pinned into the system message, and reminders of earlier turns go
  in front of a question. With `/remote on`, both go to the remote model with
  the rest of the request: the person's choice to send a turn out sends what
  SYNTHIA knows that is relevant to it too.
- The first idle consolidation after each daemon start asks only pairs never
  judged. A change of the merge prompt or model would need the remembered
  judgements (`facts_apart`) cleared, so pairs are asked again.
- Exact search costs time in proportion to the store. The HNSW index is
  built and measured for the size where a query would cost more than
  embedding it; nothing uses it yet.
