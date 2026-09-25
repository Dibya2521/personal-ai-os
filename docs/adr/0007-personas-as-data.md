# 0007. Personas are data, rendered to a fixed prompt

Status: accepted.

## Context

SYNTHIA's character should be changeable while it runs, not fixed in code.
Its default is a blend of three characters: JARVIS's precision and dry wit, a
warm companion's care, and EDITH's vigilance over security and privacy. The
person should be able to switch to one of them, nudge a single trait ("less
wit"), or write a persona of their own, and each change should apply from the
next answer without restarting anything.

## Decision

- **A persona is a TOML file**, named after the key that selects it
  (`edith.toml` is `/persona edith`). It holds a name, a one-line description,
  principles (plain sentences such as "Be honest: never invent a fact, a
  number or a source."), and either `[traits]` or `[blend]`.
- **Five traits, each a number from 0 to 1**: warmth, formality, wit,
  vigilance and verbosity. Rendering turns each into one of three fixed
  sentences: low below 1/3, high from 2/3, middle between. Wit, for example,
  is "Stay earnest; no jokes.", "Allow a light touch of humour when it fits."
  or "Use dry wit freely, never at the person's expense."
- **A blend names other personas with weights.** Its traits are their
  weighted mean, and its principles are its own followed by theirs, without
  repeats. The default `synthia` is a blend: `jarvis` 0.5, `companion` 0.3,
  `edith` 0.2, so changing an ingredient changes the default too. A blend that
  loops back on itself, or names a persona that does not exist, is refused
  with the chain that caused it.
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
pulls every trait towards the centre. The default comes out at warmth 0.535,
formality 0.59, wit 0.515, vigilance 0.61 and verbosity 0.38, which is the
middle sentence on all five, so its character comes mostly from its eight
principles (two of its own, six from its ingredients) rather than from its
traits. Letting a blend also set a
few traits over the mean would fix that; it is not built yet.
