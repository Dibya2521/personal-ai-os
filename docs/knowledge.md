# Knowledge

SYNTHIA reads the folders the person adds (notes, documents, code and PDFs)
and answers from them with citations. Each passage it finds names its file
and lines, a chat lists those citations under the answer, and a passage from
a private folder keeps the chat on this machine. Why it is built this way is
in [ADR 0013](adr/0013-knowledge.md). This page is how it works, with each
number and the command that measured it.

## What each model adds

Knowledge needs an embedding model. Its tokenizer counts the tokens files are
cut into chunks by, and the model embeds every chunk. The other models each
add a part.

| Installed | What knowledge does |
| --- | --- |
| nothing | nothing; there is no index, and every `synthia knowledge` command exits 2 and says to run `synthia models install` |
| an embedding model | reads the folders, keeps them in step, finds passages by words and meaning, and gives the chat the `search_knowledge` tool |
| also the local model | also searches deep on request, and lets a chat read private passages when the local model takes tools |
| also a reranker | nothing in the daemon, which loads no reranker; the eval scores the reranked rankings |

The daemon opens one embedding model for memory and knowledge alike, the
default all-MiniLM-L6-v2 when it is installed. A chunk fits what that model
reads: 256 tokens for all-MiniLM-L6-v2, 512 for bge-small-en-v1.5.

## Sources

A source is a folder the person adds with `synthia knowledge add <folder>`.
SYNTHIA only reads it. A folder whose root holds an `.obsidian` folder is a
`vault`, any other folder is `files`; the kind is shown, and links resolve
by the same rule in both.

- **Name.** The folder's own name, then `name-2`, `name-3` while that one is
  taken. Every citation starts with it.
- **Private.** `--private` marks a source private, `--public` undoes it. A
  new source is not private. See [Private folders](#private-folders).
- **Excludes.** Each `--exclude <path>` keeps a folder or file inside the
  source out, and the list given replaces the earlier one. An exclude is
  matched without case, so a folder kept out stays out however its name is
  typed, and a backslash separates names as a `/` does. An exclude that is
  absolute, climbs out with `..`, or is the root itself is refused.
- **Adding again.** Adding a folder that is already a source changes only
  what is given, so adding a private vault again never makes it public.
- **Refused.** A folder that does not exist, is not a folder, or is inside
  or holds another source's folder.

Nothing is read until the next pass (see [Keeping in step](#keeping-in-step)).
`synthia knowledge remove <folder>` forgets the source and everything read
from it: documents, chunks, vectors, links and words, in one transaction.

## Which files are read

A file's suffix, matched without case, picks its reader. Any other suffix is
passed over unopened, so a vault's images and attachments are never read.

| Kind | Suffixes |
| --- | --- |
| Markdown | `.md`, `.markdown` |
| Python | `.py`, `.pyi` |
| Plain text | `.txt`, `.rst`, `.csv`, `.toml`, `.yaml`, `.yml`, `.json`, `.ini`, `.cfg`, `.sql`, `.sh`, `.ps1`, `.ts`, `.tsx`, `.js`, `.mjs`, `.css`, `.html` |
| PDF | `.pdf` |

The walk passes over dot folders and dot files (so `.obsidian`, `.git` and
an Obsidian `.trash` are never read), excluded paths, links and junctions,
which are never followed, and names that are not valid text.

Text is decoded strictly, and its lines are numbered as an editor numbers
them. It must be UTF-8;
a byte order mark is dropped; a NUL byte in the first 8,192 bytes marks the
file as binary; and a line ends at `\r\n`, `\r` or `\n`. A PDF is read page
by page with pypdf, and lines are counted within each page.

Some files are kept with no chunks and a note saying why, so they are not
read again until they change:

- **Too large.** A file over 2 MiB, about four times the largest note in a
  large Obsidian vault and eight times the largest file in this repository.
- **Unreadable.** Binary, not UTF-8, or a file a reader failed on. A PDF
  that is encrypted is noted `encrypted: not read`, even when an empty
  password opens it, and one pypdf cannot parse is noted with pypdf's
  reason.
- **Held back.** A file that looks like it holds a secret. A plain-text file
  is held back by its name alone when it matches `*secret*`, `*credential*`,
  `*password*`, `*.pem`, `*.key`, `id_rsa*`, `*.pfx` or `*.p12`. Any file is
  held back by its text when a line holds a private key block, an AWS access
  key id, a GitHub token, an `sk-` key, a URL with a password in it, or a
  name such as `api_key`, `secret`, `token` or `password` set to 16 or more
  characters that are not a placeholder. Unquoted, such a value must hold a
  digit, must not be digits alone, and must hold no `_` or `.` (the shape of
  a name in code), so `token = settings.api_token` and
  `token_budget=1000000000000000` are code, not secrets. The rules err on
  holding back, and they miss a secret in another shape, such as a bare JWT
  or a key under another name.

A PDF page with no text, which is what pypdf returns for a scanned page, is
named in the document's note ("pages 3, 5 hold no text (scanned or images
only)"). A PDF with no text at all is kept with that note and no chunks.

## From a file to chunks

A file goes through three steps, each with one job.

1. **Decode.** The bytes become pages of lines, as above.
2. **Read.** A reader returns an outline: the title, blocks of lines under a
   trail of headings, and the links on each line. It never returns text, so
   every chunk is exactly the file's lines.
3. **Cut.** The chunker cuts any reader's outline into chunks that fit the
   embedding model.

### Readers

- **Markdown.** The title is the frontmatter's `title`, else the first H1,
  else the file name without its suffix. A `#` line is a heading only outside
  a fence. Frontmatter is read for `title` and for wikilinks under `related`
  and `patterns`, not as YAML. Links are wikilinks (`[[Note]]`, with a folder
  path, a heading anchor or an alias), embedded notes (`![[Note]]`) and
  relative Markdown links. Images, embedded attachments, anything inside a
  fence or inline code, and a link that climbs above the source's root are
  not links.
- **Python.** Read with the standard library's `ast`. The title is the module
  name Python imports the file by (a leading `src/` dropped, `__init__`
  naming its package). Each top-level function and class is a block that
  starts at its first decorator or at the comment lines right above it; a
  class is its head, then each method. Every import becomes a link. `from
  a.b import c` links `a.b` and `a.b.c`, a relative import resolves against
  the file's package, and a bare standard library name such as `import
  types` is no link. A file that does not parse is read as plain text, and
  its note says why.
- **Plain text and PDF.** Each page that holds text is one block, under the
  file's name.

### Chunks

The chunker counts each line once with the embedding model's own tokenizer.
That tokenizer splits on whitespace and reads a line break as a space, so a
span of lines costs exactly the sum of its lines, and packing is arithmetic.

The default cut is windows of 40 lines with no header. Each window takes the
next 40 lines, drops the blank lines at its ends, and gives up lines from its
end until it fits the model's limit, less its start and end tokens. A single
line longer than that is a chunk of its own, marked cut, and the embedding
model reads only its start. No chunk crosses a PDF page. Under the default, a
reader's blocks do not decide where chunks end; the reader still gives each
document its title, its links and its note.

A second cut, by structure, is kept for the eval to compare. It keeps a
reader's unit whole when it fits, else each block, else packs paragraphs
while they fit, else takes lines. With a header on, each chunk starts with
its trail ("Note > Section > Subsection") in at most a quarter of the limit.
On this repository's questions it scored below the windows (see
[Numbers](#numbers)).

Each document records the recipe it was cut by, such as
`lines/256/40/r1/s2`: the cut, the token limit, the window, the readers'
version and the secret rules' version. A pass re-reads every document whose
recipe differs, so a change to any of them takes effect everywhere.

### Citations

A passage cites itself as `name/path:start-end`, where `name` is the
source's name, `path` the file's path from the source's root, and the lines
are the editor's, both ends included. A single line is `name/path:N`. A PDF
passage names its page first, as `name/path:p.2:4-9`; pages are numbered
from 1 even in a one-page PDF, so a PDF citation always names its page.

### Links

Which document a link reaches is never stored. It is worked out each time it
is read, within the link's own source, so a note added or renamed later is
found by links written before it, with no pass over the links.

- **Wikilinks** resolve in the order Obsidian 1.14 uses. A link with folders,
  such as `[[Archive/Plan]]`, reaches the note at that path from the root,
  else a note whose path ends with it, notes under the linking note's folder
  first, then the shorter path. A bare name reaches the note of that name
  under the linking note's folder first, then anywhere, shorter path first.
  A note comes before another file of the same name. The trailing match stops
  at a `/`, so `[[chive/Plan]]` never reaches `Archive/Plan.md`.
- **Markdown links** reach the document at that path.
- **Imports** reach the module of that name, else one whose dotted name ends
  with it, shorter path first.

A document never links to itself.

## Keeping in step

A pass walks every source and compares each file with what the index holds,
so it decides from the files and the index alone, and a pass that stopped or
crashed anywhere is mended by the next one.

- A file whose size, modification time and recipe match what was stored is
  not opened.
- A file whose time changed but whose bytes did not (the same SHA-256) only
  has its new time recorded.
- Any other file is read, and its chunks are replaced whole in one
  transaction, so a failure part way leaves the old document as it was.
- A file that could not be opened this time is passed over and tried again
  on the next pass.

Nothing is removed on missing evidence. A source whose folder cannot be
listed is reported unavailable and keeps its documents, and so does a source
that lists empty while documents are stored, which is what an unmounted
share or a folder not yet synced looks like. A sub-folder that cannot be
listed keeps its documents too. Only a file absent from a folder that was
listed is removed.

A pass that wrote anything then merges the full-text index's segments and
refreshes SQLite's query planner statistics. Chunks are embedded last, 64 at
a time, taken from the chunks the index holds without a vector, so chunks
written before a crash are embedded by the next pass.

Adding a source, removing one and a pass take one lock, so a removal that
arrives during a pass waits for it instead of deleting what the pass is
writing. A search takes no lock, since SQLite's write-ahead log lets it read
beside a writer.

`synthia knowledge index` runs a pass now and prints its counts in one line:
files added, changed, unchanged, removed, unreadable, held back and passed
over, then chunks written, cut and embedded, and the seconds taken. The
daemon also runs passes on its own (see [In the daemon](#in-the-daemon)).

## Finding passages

A search ranks every chunk one of three ways.

- **Words.** BM25 from SQLite's full-text index (FTS5, Porter stemming) over
  each chunk's header and body, matching any word of the query.
- **Meaning.** The cosine similarity of the query's vector to every chunk's
  vector. The vectors are held in memory and compared in one matrix product,
  an exact search, as [memory's](memory.md#finding-earlier-turns) is.
- **Fused,** the default. Both, in one score:

```text
cosine similarity + 0.2 * bm25 / (ln(2N + 1) * (k1 + 1))
```

N is the number of chunks in the index and k1 is 1.2, so the divisor is the
most one query word can score. Words put forward their best 50 chunks and
meaning its closest 50 (or as many as the search asks for, if more), and
every chunk the words found also gets its cosine. Ties go to the lower chunk
id. Memory fuses with the same formula at weight 0.1, measured on its own
questions; knowledge's 0.2 was measured on its own (see [Numbers](#numbers)).

A query with no word in it, only punctuation, finds nothing.

### Variants

`synthia knowledge search --variant` and the daemon's `knowledge_search`
take a ranking by name. Each name reads back exactly as written.

| Name | Ranks by |
| --- | --- |
| `words` | words alone |
| `meaning` | meaning alone |
| `fused` | the default, at word weight 0.2 |
| `fused:<weight>` | fused at another word weight, such as `fused:0.1` |
| `<ranking>+hop1x<weight>`, `+hop2x<weight>` | then follows links one or two steps |
| `<ranking>+rerank:<reranker>@<N>` | then reorders the first N with a reranker; last in the name |

**Following links.** A hop takes the best 5 chunks, follows their resolved
links, and gives every chunk of each document reached its own score for the
query plus the link weight times the best score among the chunks linking to
it. A chunk already found keeps the higher of its two scores. A second hop
starts from the best chunks the first reached, so a chain `a -> b -> c`
carries `a`'s score to `c`.

**Reranking.** A cross-encoder, ms-marco-MiniLM-L6-v2, reads the question
and each passage together and scores the pair, so it can reorder the first
passages a search finds; the reordered passages carry its score. It comes
as `ms-marco-minilm-l6-int8` (8-bit, 22.8 MiB with its tokenizer) and
`ms-marco-minilm-l6` (full precision, 87.5 MiB), and neither is installed
unless named. The 8-bit file scores a pair differently with other pairs
beside it, so it runs one pair at a time; the full-precision file runs 16
at once. The daemon loads no reranker, so it refuses a reranked variant;
the eval scores them.

Neither following links nor reranking is the default, since on this
repository's questions neither ranked reliably better within a 2-second
search (see [Numbers](#numbers)).

### Deep search

A question asked in other words than the file that answers it shares few of
its words, and the embedding model alone does not close the gap. Deep search
asks the local model, in one request (temperature 0, thinking off), for up to
3 search queries in the words the files likely use (at most 200 characters
each) and a passage of three to six sentences written as if quoted from a
file that answers the question (at most 1,200 characters). The question, each
query and the passage are searched alike, and each chunk keeps its best
score over them. The eval also measures merging them by reciprocal rank (the
sum of 1 / (60 + rank)).

- **On request only.** It costs a generation of tens of seconds on a CPU, so
  it runs only for `synthia knowledge search --deep` and `/deep <question>`
  in the chat. The `search_knowledge` tool the model sees has the same one
  parameter either way, so the model can never start one.
- **Local only.** The request does not allow the remote model, so the router
  sends it to the local model, and the question never leaves the machine.
  With no local model installed, deep search fails and says so.
- **Checked first.** The variant is checked before the model is asked, so a
  wrong one costs no generation. A reply that never fits the asked shape,
  after the gateway's repairs, leaves the plain search's passages and a note
  saying so.

`--deep` prints one line before the passages: how many rewrites and whether
a passage were written, the seconds spent asking the local model and the
seconds spent searching.

## In a chat

The chat model has the tool `search_knowledge`. It reads only this machine's
index and changes nothing, so it runs without asking. It takes one
parameter, what to look for, and returns up to 6 passages, best first. Each
comes under its citation in square brackets and its section or document
title, its text cut at 2,400 characters (at the last line end when that
keeps at least 1,200, otherwise mid-line), and, when its links reach other
documents, `links to:` and their titles, so the model can search for one
next. Six passages stay near 15,000 characters, which leaves room for the
answer in a 32,768-token window. When nothing matches, the result says
"nothing found in the person's files".

Every citation a turn's searches showed is kept, first shown first and each
once. The turn's report carries them as `sources`, and `synthia chat` prints
them in a dim line under the answer, so the model does not have to write
them itself.

`/deep <question>` first runs one deep search for the question through
`search_knowledge`'s own code, as if the model had called it, so the privacy
rules, the sources and the trace are those of any search. The model then
reads the result as its own call, may search again plainly, and answers.

## Private folders

A private passage is shown only where it cannot leave the machine.

- **In a chat.** A passage from a private source is shown only to a chat
  whose turn is running with a local model that takes tools behind it. The
  chat is confined before the tool returns, so the turn's next request is
  already local, and the router sends every request of that chat to the
  local model until `/reset`, whatever `/remote` says. Anywhere else private
  passages are withheld, and the result says how many ("2 private passages
  were withheld: this chat cannot stay on this machine"), so a question only
  private text answers is not mistaken for nothing found. The count is of
  the private passages among the best 6.
- **Through links.** A link reaches only documents of its own source, so a
  public passage never names a private document.
- **In memory.** The turns of a confined chat are remembered marked, as
  [memory's Private text](memory.md#private-text) describes.
- **In the terminal.** `synthia knowledge search` shows private passages,
  marked `(private)`, since the terminal is the person's own.

## In the daemon

The index is one SQLite file, `SYNTHIA_HOME/knowledge.db`, beside
`memory.db`. Everything in it can be read again from the folders, and
nothing in memory can, so a rebuilt or damaged index never touches memory.
It follows memory's rules: write-ahead logging, a connection per call, every
write in one transaction, all work off the event loop, and a full-text index
kept in step by triggers. Every `synthia knowledge` command goes through the
daemon, starting it if needed, so one process writes the index. Its methods
are in [`daemon.md`](daemon.md#messages).

When no turn has run for 10 minutes, at most once an hour after the last
pass ended, the daemon runs a pass on its own, so the index follows files
that changed. Such a pass stops before its next file, or its next round of
embedding, once a turn starts or another `synthia knowledge` command needs to
change the index, so neither waits for it. What it read is kept, and the
next 10 idle minutes go on from there. A pass asked for with
`synthia knowledge index` runs through, since the person is waiting for it.
Every pass, asked or not, stopped, failed or cancelled, is a record in the
daemon's trace.

## Commands

| Command | What it does |
| --- | --- |
| `synthia knowledge add <folder>` | adds a folder, or changes one added before: `--private` or `--public`, and `--exclude <path>` as often as needed |
| `synthia knowledge remove <folder>` | stops reading a folder and forgets everything read from it |
| `synthia knowledge list` | shows each folder: name, kind, private or not, where it is, what it leaves out |
| `synthia knowledge index` | runs a pass now and prints its counts; names any folder that could not be listed; exits 2 when the pass failed; Ctrl+C stops it and keeps what was read |
| `synthia knowledge search "<question>"` | prints the best passages, each with its citation, its score, `(private)` when private, and its first 8 lines; `--limit` (1 to 50, default 6), `--variant`, `--deep` |
| `/deep <question>` in `synthia chat` | searches deep first, then answers |
| `synthia trace` | shows the daemon's passes of the index among its records |

## Numbers

Each was measured in October 2026 with the command beside it, on a laptop CPU
while other work kept it busy. Timings move with the load and the machine;
the scores do not, since embedding and ranking are deterministic (a second
full run of the first eval gave identical scores).

Search is scored on `benchmarks/data/knowledge_set.json`: 61 questions about
this repository at commit `ffd814e`, written before any search ran so none
could be tuned to them. 12 name a rare symbol or setting, 12 ask in other
words than the answer uses, 13 ask where something is done in the code, 12
need two places, and 12 ask about something no file answers. The 49
answerable ones are answered by 71 line ranges in 43 files. A passage covers
a range when it is in that file, on the same page, and shares a line with
it, and each range counts once, at the first passage that covers it.
nDCG@10 counts 1 for each of the first 10 ranks that covers a new range,
divided by log2(rank + 1), and divides that sum by the sum of an ideal
ranking that covers one range at each of the first ranks. Each table gives
its mean over the answerable questions. "Better in X %" is how often a
ranking scored higher in 2,000 paired bootstrap resamples over the
questions.

**Chunking and ranking** (`uv run python benchmarks/rag_eval.py`). The
repository's 261 files at that commit were indexed once per chunking, and
each question was searched 10 passages deep. nDCG@10 by chunking and
ranking:

| Chunking (chunks) | words | meaning | fused:0.05 | fused:0.1 | fused:0.2 | fused:0.4 |
| --- | --- | --- | --- | --- | --- | --- |
| sections with a header (3,906) | 0.378 | 0.332 | 0.358 | 0.390 | 0.437 | 0.440 |
| sections (3,776) | 0.360 | 0.310 | 0.338 | 0.369 | 0.391 | 0.389 |
| 40-line windows (2,085) | 0.380 | 0.402 | 0.423 | 0.442 | 0.461 | 0.430 |

Windows at weight 0.2, the default, beat the earlier default (sections with a
header at weight 0.1) in 98.1 % of resamples, by 0.015 to 0.128 (90 %
interval). Weight 0.2 beat 0.1 in 99.7 % with sections and 86.5 % with
windows. Windows beat sections with a header at 0.2 in only 76.6 %; windows
are the simpler cut, so they are the default until a better one is measured.
A likely reason sections lose is that they are never packed with their small
neighbours, so the same files gave 3,906 section chunks against 2,085
windows. Questions asked in other words scored near zero under every ranking
(best 0.227, meaning over windows; words 0, since those questions were
written without the files' own words), so the embedding model, not the
ranking, limits them. Of the 1,098 searches, the slowest 5 % took under
150 ms, query embedding included.

**Following links** (the same command). The default against itself with
hops, over 40-line windows:

| Ranking | nDCG@10 | Better than fused | Median ms a search |
| --- | --- | --- | --- |
| fused | 0.461 | | 68 |
| fused+hop1x0.1 | 0.454 | 24 % | 127 |
| fused+hop1x0.3 | 0.443 | 17 % | 132 |
| fused+hop2x0.3 | 0.439 | 12 % | 173 |

None beats the default more than 24 % of the time, so the default follows no
links. The 12 questions whose answer spans a doc and a code file have no link
between the two files, because the docs name code in backticks, not as links.

**Reranking** (the same command, with both rerankers installed). The first N
passages of the fused ranking, reordered. Each table is its own run, and
times move with the machine's load: the same fused search had a median of
68 ms in the run above and 45 ms in this one, with identical rankings.

| Ranking | nDCG@10 | Median ms | 95th percentile ms |
| --- | --- | --- | --- |
| fused | 0.461 | 45 | 58 |
| 8-bit, first 10 | 0.423 | 1,337 | 1,778 |
| 8-bit, first 20 | 0.477 | 2,138 | 3,683 |
| full precision, first 10 | 0.423 | 1,693 | 2,375 |
| one hop at 0.3, then 8-bit, first 10 | 0.426 | 1,466 | 2,512 |

Reranking the first 20 is better in only 63 % of resamples and takes 3.7 s at
the 95th percentile. Code questions get worse under every reranked ranking
(better in at most 4 %), since ms-marco was trained on web passages, not
code. A pair of a question and a 40-line passage costs about 110 ms, so only
about 10 to 15 pairs fit a 2-second search. The 8-bit file run in batches moved a
pair's score by up to 0.24 against the same pair alone, which is why it runs
one pair at a time.

**Deep search** (`uv run python benchmarks/deep_expand.py`, then
`benchmarks/rag_eval.py --expansions`). Each question was expanded once by
the local qwen3.5-4b and searched over 40-line windows. A generation took
35.7 s at the median (19.8 to 136.0 s), so deep search can never fit a
2-second search and is never the default.

| Searched beside the question | nDCG@10, best score | nDCG@10, reciprocal rank |
| --- | --- | --- |
| nothing (fused) | 0.461 | |
| the rewrites | 0.461 | 0.468 |
| the drafted passage | 0.486 | 0.474 |
| both | 0.486 | 0.479 |
| both, then 8-bit reranking of the first 10 | 0.496 | 0.495 |

No form is reliably better over all questions (better in at most 81 %). By
kind, questions naming a rare symbol gain with the passage (+0.084, better in
98.5 %), questions in other words gain most with both merged by rank
(+0.115, better in 84.9 %), and code questions never gain and are worse
reranked. Merged by best score, both score exactly as the passage alone,
because a drafted passage reads like a file, so its similarities top every
query's.

**Indexing and the daemon** (`synthia knowledge index` and
`synthia knowledge search`, on this repository in a fresh home with the
8-bit all-MiniLM-L6-v2). A first pass read 269 files into 2,215 chunks in
262.8 s and held back 12 test files for their realistic fake keys. A second
pass, with 3 files changed, took 5.4 s. Measured on the indexer alone before
the daemon held it, a pass over 263 files with none changed took 0.16 s. A
`knowledge_search` request from a new client, for 40 questions of the set,
took 1,007 ms at the median and 1,140 ms at the 95th percentile. About
880 ms of it is finding the running daemon, which every command pays.

**The chat's answers** (`uv run python benchmarks/chat_answers.py`, then
`benchmarks/answer_eval.py --no-judge`). Every question was asked once
through the chat with the local qwen3.5-4b, `search_knowledge` its only
tool, and no remote model, 102.5 minutes of chat in all. "Searched" is the
share that called the tool, and "shown" the share of answerable questions
where a passage covering an answer's range was in front of the model.

| Questions | Searched | Shown | Median s | 95th percentile s |
| --- | --- | --- | --- | --- |
| rare symbol (12) | 0.667 | 0.500 | 76.6 | 226.1 |
| other words (12) | 0.583 | 0.167 | 98.2 | 139.8 |
| code (13) | 0.692 | 0.462 | 106.4 | 177.1 |
| two places (12) | 0.917 | 0.750 | 106.6 | 212.7 |
| no answer (12) | 0.500 | | 62.9 | 307.1 |
| all (61) | 0.672 | 0.469 | 100.3 | 177.1 |

No turn ended in an error. The model answered 20 of the 61 without searching,
4 of them questions that name the exact identifier asked about, so those
answers come from its own memory. No answer that searched writes a citation
in its text, since the chat lists the sources under each answer itself.
Whether each answer keeps to its passages needs a judge checked against a
person's labels first, so it is not reported yet.

**PDF** (a generated 50-page PDF of 40 lines a page). Reading it took 10.7 ms
a page, best of 5. Importing pypdf took about 1.3 s, more than the whole
indexer module (0.88 s), so it is imported only when the first PDF is read.
