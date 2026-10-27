# 0013. Knowledge: the person's folders in their own SQLite file, found by words and meaning, private passages kept on this machine

Status: accepted.

## Context

Before knowledge, SYNTHIA could read a file only when the model asked for it
by path, inside the folders named in `SYNTHIA_FILE_ROOTS`. It could not
answer "where is the token rate learned?" unless the person already knew
where the answer was. The goal is that SYNTHIA answers from the person's own
notes, documents and code, and cites the file and the lines each part of the
answer came from.

Facts that shaped the design:

- Everything runs on the person's machine first (ADR 0004, 0009), usually a
  laptop CPU that is also doing the person's other work. One daemon holds
  SYNTHIA (ADR 0011), and memory already keeps its turns in SQLite with words
  and meaning searched together (ADR 0012).
- An index of files can be rebuilt from the files at any time. Memory cannot
  be rebuilt from anything.
- The code memory searches with (the embedding model runner, its tokenizer,
  exact vector search, the HNSW index, BM25 and the fused score) lived in the
  `memory` package, which knowledge is not part of.
- A citation can be checked only if a chunk is exactly a run of the file's
  lines, numbered as an editor numbers them.
- Some of the person's folders are private. Text read from them must never
  reach the remote model, whether in the request that read it, in the later
  steps of the same turn, or later through memory. The agent copies a turn's
  first request into every later step, so a check made only where the first
  request is built would let the rest of the turn leave.
- The local model is slow on a CPU. In October 2026 one request to rewrite a
  question took 35.7 s at the median and up to 136.0 s
  (`benchmarks/deep_expand.py`), and a search is meant to answer within 2
  seconds.
- No choice of chunking or ranking was made before it was measured. Each was
  scored on `benchmarks/data/knowledge_set.json`, 61 questions about this
  repository whose answers are known line ranges, written before any search
  ran.

## Options

**Where the index lives.**

1. **In `memory.db`.** One file, and a rebuild or a damaged index would
   touch what cannot be rebuilt.
2. **A vector database server.** It lost for the reasons ADR 0012 gives.
3. **Its own SQLite file, `knowledge.db`,** following memory's rules.

**Where the shared search code lives.**

1. **Knowledge imports it from `memory`.** The smaller first change, but
   knowledge would depend on a package it is not part of, and the reranker
   would have to go in one of the two.
2. **A `retrieval` package below both**, which neither owns, with knowledge
   and memory forbidden to import each other.

**How the index follows changed files.**

1. **A file watcher (watchdog).** A new dependency and a watch per folder,
   for edits an idle pass catches within minutes.
2. **Only when asked.** The index goes stale unless the person remembers.
3. **A pass when the daemon is idle**, deciding from the files and the
   index alone, plus a pass on request.

**How code is read.**

1. **tree-sitter.** Binary wheels, and grammars for languages no source
   holds yet.
2. **The standard library's `ast`.** Exact for Python, the language with
   real volume here, and no new dependency. Other text is read by lines.

**How files are cut.** Each was measured with `benchmarks/rag_eval.py`
(nDCG@10 over the 49 answerable questions, October 2026).

1. **Sections with a header**, each chunk starting with its heading trail.
   It scored 0.390 at word weight 0.1 and 0.437 at 0.2. Sections are never
   packed with their small neighbours, so the same 261 files gave 3,906
   chunks.
2. **Sections without a header.** It scored 0.391 at most.
3. **Windows of 40 lines without a header.** It scored 0.461 at word weight
   0.2, from 2,085 chunks.

**How passages are ranked.** Each was measured the same way.

1. **Words alone.** 0.380 at best, and 0 on the questions asked in other
   words than the answer.
2. **Meaning alone.** 0.402 at best.
3. **Words and meaning fused**, memory's score, at weights 0.05, 0.1, 0.2
   and 0.4. 0.461 at 0.2, which beat 0.1 in 86.5 % of 2,000 paired
   bootstrap resamples over the questions with windows and 99.7 % with
   sections.
4. **Following links** from the best 5 passages. 0.454 at best, and never
   better than the plain fused ranking in more than 24 % of resamples. The
   questions whose answer spans a doc and a code file have no link between
   the two, since the docs name code in backticks.
5. **A cross-encoder reranker** over the first 10 or 20 passages. 0.423 at
   10; 0.477 at 20, but better in only 63 % of resamples and 3.7 s at the
   95th percentile. Code questions got worse under every reranked ranking
   (better in at most 4 %), since ms-marco was trained on web passages.
6. **The local model rewriting the question** and drafting an answer to
   search for too. 0.486 at best and 0.496 reranked, better in at most 81 %
   of resamples, at 35.7 s a question.

**How vectors are searched.**

1. **SYNTHIA's own HNSW index.** Measured in ADR 0012, and a second
   structure to keep in step with every removal.
2. **Exact search** over vectors held in memory, as memory does. Memory
   measured 6.1 ms at 50,000 vectors, about what embedding the query costs.

**Where a link's target is kept.**

1. **Stored when the link is read.** A note added or renamed later is missed
   until every link to it is read again.
2. **Worked out when read**, by a view over the documents.

**How private passages are kept on the machine.**

1. **Never shown in a chat.** Safe, but a person's private notes could then
   answer nothing in a chat.
2. **A check where a turn's first request is built.** The later steps of the
   turn copy that request, so they would still leave.
3. **A latch per chat that the router checks on every request.** Whatever
   reads private text closes it, and every later request of the chat goes
   local until `/reset`.

## Decision

- **Its own SQLite file**, `knowledge.db`, in write-ahead logging mode, with
  memory's rules: a connection per call, every write in one transaction, all
  work off the event loop, a full-text index kept by triggers, and removals
  by cascade. Chunk ids are never reused, so a vector held in memory is never
  taken for a newer chunk's.
- **A `retrieval` package** holds the code both stores search with, and the
  reranker. Knowledge and memory sit side by side above it, and the import
  check forbids either to import the other.
- **Readers return line spans, never text.** Markdown is read in one pass,
  Python with `ast`, and plain text and PDF pages by lines. Every chunk is
  exactly the file's lines, cited as `name/path:start-end`, with `p.N:` for
  a PDF page.
- **The default chunking is windows of 40 lines without a header**, at the
  embedding model's token limit. It beat sections with a header in only
  76.6 % of resamples at the same weight. It is the simpler cut, so it stays
  the default until a better one is measured.
- **The default ranking is fused at word weight 0.2.** Memory keeps 0.1,
  measured on its own questions. Link hops, the reranker and the rewrites
  stay available by name, and the eval scores them on every run. The daemon
  loads no reranker.
- **Exact vector search**, the vectors held in memory and loaded when the
  index opens.
- **The index follows the files by passes.** A pass compares each file's
  size, time and recipe with what is stored, reads only what changed,
  replaces a changed document whole, and removes a document only when its
  folder was listed without it. The daemon runs one after 10 minutes without
  a turn, at most once an hour, and it gives way to a turn or a waiting write
  before its next file.
- **Link targets are worked out when read**, wikilinks in the order Obsidian
  1.14 resolves them, within the link's own source.
- **Files that look like they hold a secret are held back unread**, by name
  for plain-text files and by text for every file. The rules err on holding
  back.
- **A private passage confines the chat.** It is shown only to a chat that
  has a local model taking tools behind it, and the router's latch then
  keeps every request of that chat local until `/reset`. Anywhere else
  private passages are withheld and counted.
- **Deep search runs only on request**, through the local model, never the
  remote, and the chat model cannot start it.

## Consequences

- A chat can answer from the person's files. The model calls
  `search_knowledge`, and the citations it was shown are listed under the
  answer. On the 61 questions in October 2026, the local model searched for
  67 % of them, and for 47 % of the answerable ones a passage covering the
  answer was in front of it (`benchmarks/chat_answers.py`, then
  `benchmarks/answer_eval.py --no-judge`). It answered the rest without
  searching, or searched and missed.
- Knowledge needs an embedding model. Without one there is no index, and
  every `synthia knowledge` command says so.
- The first pass over a folder takes minutes. This repository took 262.8 s
  for 269 files on a busy laptop CPU. A later pass reads only what changed.
- Questions asked in other words than the answer remain weak, at a best
  nDCG@10 of 0.227 without deep search. Deep search helps them most and
  costs tens of seconds, so it waits for the person to ask.
- A change of chunking, of a reader or of the secret rules re-reads every
  file on the next pass, since each is part of every document's recipe.
- With `/remote on`, a public passage goes to the remote model with the rest
  of the request, as memory's reminders do. A private passage never does.
- Exact search costs time in proportion to the index, as in memory. The HNSW
  index stays measured and unused until an index outgrows exact search.
