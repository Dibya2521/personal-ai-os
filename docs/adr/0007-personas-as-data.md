# 0007. Personas are data, rendered to a fixed prompt

Status: accepted.

## Context

SYNTHIA's character should be changeable while it runs, not fixed in code.
There is one SYNTHIA; its personas are variants under other names. Three are
pure characters: Nova (casual, sassy, tactical), Horizon (security and
privacy first) and Zenith (formal and precise). The default mixes all three,
mostly Nova: a caring, quick-witted companion, careful with data. Neon,
Glacier and Starlight each let one of the three lead. Two characters stand
alone: Minato (inspired by Minato Namikaze in Naruto), a calm, humble mentor
with nerves of steel and a kind heart, and Yume (inspired by Kaoruko Waguri
in The Fragrant Flower Blooms with Dignity), a gentle companion for
conversation who sees people by what they do. The person should be able to switch between
them, nudge a single trait ("less wit"), or write a persona of their own, and
each change should apply from the next answer without restarting anything.

## Decision

- **A persona is a TOML file**, named after the key that selects it
  (`nova.toml` is `/persona nova`). It holds a name, a one-line description,
  principles (plain sentences such as "Be honest: never invent a fact, a
  number or a source."), and either all five `[traits]` or a `[blend]`.
- **Five traits, each a number from 0 to 1**: warmth, formality, wit,
  vigilance and verbosity. Rendering turns each into one of three fixed
  sentences: low below 1/3, high from 2/3, middle between. Wit, for example,
  is "Stay earnest; no jokes.", "Allow a light touch of humour when it fits."
  or "Use dry wit freely, never at the person's expense."
- **Warmth is a resting tone, not a fixed one.** Each warmth sentence lets the
  moment move it either way: low is "Keep a cool, matter-of-fact tone, and
  warm up when the moment calls for it.", high ends "and turn cool and
  matter-of-fact when the moment calls for it." A warm persona can still be
  cool when that is what the situation needs, and a cool one can comfort.
- **A blend names other personas with weights.** Its traits are their
  weighted mean, and its principles are its own followed by theirs, without
  repeats. A blend may also give some `[traits]`, which replace the mean for
  those sliders only.
- **The default `synthia` is `nova` 0.5, `horizon` 0.3, `zenith` 0.2.** Its
  mean gives formality 0.425 (plain and polite) and vigilance 0.83 (active),
  and it pins warmth at 0.8, so editing an ingredient never cools it, wit at
  0.7 for Nova's sass, and verbosity at 0.5, because Nova's brevity ("as few
  words as will do") would cut a companion's explanations short.
- **Led mixes give their lead 0.7 and each other character 0.15**: `neon`
  (Nova), `glacier` (Horizon), `starlight` (Zenith). They pin nothing, so
  each is its lead's character softened by the other two; Glacier renders the
  same five sentences as Horizon and differs in principles.
  A blend that loops back on itself, or names a persona that does not exist,
  is refused with the chain that caused it.
- **Files in `SYNTHIA_HOME/personas` replace built-ins with the same key**, so
  a person can redefine even the default without touching the package. Files
  saved with a byte order mark (as Windows Notepad can) are read as well.
- **The system prompt is a `string.Template`** filled with the name,
  description, trait sentences and principles.
- **In the chat**, `/persona <name>` switches and `/persona set wit=0.3`
  moves sliders; the conversation so far is kept and the next answer uses the
  new persona. `SYNTHIA_PERSONA` picks the one to start in.

## Options considered

- **One fixed prompt in code.** Simplest, but every change of character is a
  code change, and nothing can be adjusted while talking.
- **Free-text prompts only.** Flexible, but "a bit less wit" becomes a rewrite,
  and there is nothing to blend.
- **Pass the slider numbers to the model** ("wit: 0.3"). Keeps every value,
  but a model is not calibrated on what 0.3 means; a sentence says what to do.
  Fixed sentences also make the prompt a pure function of the persona, so it
  is tested exactly and identical prompts can be cached.
- **Five bands per trait instead of three.** Finer, but each band is another
  sentence to write and test for every trait. Three are kept until they prove
  too coarse.
- **`str.format` templates.** A format string can reach into the attributes of
  the values it is given (`{name.__class__}`), and templates come from user
  files. `string.Template` substitutes names and nothing else.

## Consequences

A new persona is a file, not a release. The weighted mean has a known cost: it
pulls every trait towards the centre. An earlier default, an even-handed
0.5/0.3/0.2 mean of three contrasting characters, came out at warmth 0.535,
formality 0.59, wit 0.515, vigilance 0.61 and verbosity 0.38, the middle
sentence on all five, which reads as no character at all. Two things answer
that: a mix gives one character the largest share, and a blend can pin
traits over its mean where an exact value matters; pinned traits do not move
when an ingredient is edited, blended ones do. Some blended values sit near
a band edge (Glacier's warmth is 0.325, just under 1/3), so the tests pin
every persona's rendered sentences, and an edit that flips a band fails
them. The default renders 10 principles: 2 of its own, 4 from `nova`, 2 each
from `horizon` and `zenith`.
