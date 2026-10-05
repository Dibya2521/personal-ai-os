"""Sentences to vectors, with an installed embedding model, on this machine.

The ONNX network runs on the CPU through onnxruntime. Texts are tokenized
with SYNTHIA's own WordPiece, sorted by length and run in batches, so a
batch pads to its longest text rather than to the longest of all; each
vector comes back in the order its text was given. The per-token outputs
become one vector the way the model was trained (the [CLS] token, or the
mean over the text's tokens) and are scaled to length 1, so the dot product
of two vectors is their cosine similarity.

Embedding blocks for as long as the network runs (a fraction of a second
for a few sentences), so :meth:`TextEmbedder.embed` runs it off the event
loop.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Final

import numpy as np
import onnxruntime as ort

from synthia.kernel.errors import SynthiaError
from synthia.memory.wordpiece import TokenizerError, WordPiece
from synthia.models.catalogue import Pooling

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from synthia.models.catalogue import Embedder

type Feeds = dict[str, NDArray[np.int64]]
type Network = Callable[[Feeds], NDArray[np.float32]]
"""Token ids, attention mask and (if the model takes them) token types in;
one row of numbers per token out, shaped (texts, tokens, dimensions)."""

BATCH: Final = 16
PADDING_ID: Final = 0
# BERT's segment ids: which sentence of a pair each token belongs to.
SEGMENT_IDS: Final = "token_type_ids"


class EmbedError(SynthiaError):
    """An embedding model that cannot be loaded, or is not the one catalogued."""


def threads() -> int:
    """Return the CPU threads to give the network: half, so the chat keeps the rest."""
    return max(1, (os.cpu_count() or 2) // 2)


class TextEmbedder:
    """An embedding model, ready to turn texts into unit vectors."""

    def __init__(
        self,
        spec: Embedder,
        tokens: WordPiece,
        network: Network,
        *,
        token_types: bool,
    ) -> None:
        """Embed with ``network``, fed ``tokens``' ids, as ``spec`` describes.

        Raises:
            EmbedError: If the network does not give ``spec.dimensions`` numbers.
        """
        self.spec = spec
        self._tokens = tokens
        self._network = network
        self._token_types = token_types
        width = self._pooled([tokens.ids("a", spec.max_tokens)]).shape[1]
        if width != spec.dimensions:
            message = f"{spec.id} gives {width} numbers, not {spec.dimensions}"
            raise EmbedError(message)

    @classmethod
    def load(cls, directory: Path, spec: Embedder) -> TextEmbedder:
        """Load ``spec``'s installed files from ``directory``.

        Raises:
            EmbedError: If a file is missing, unreadable or not the model the
                catalogue describes.
        """
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads()
        try:
            tokens = WordPiece.from_file(directory / spec.tokenizer.name)
            session = ort.InferenceSession(
                str(directory / spec.model.name),
                options,
                providers=["CPUExecutionProvider"],
            )
        except (OSError, RuntimeError, TokenizerError) as error:
            message = f"cannot load the embedding model {spec.id}: {error}"
            raise EmbedError(message) from error

        def network(feeds: Feeds) -> NDArray[np.float32]:
            return np.asarray(session.run(None, feeds)[0], np.float32)

        names = {i.name for i in session.get_inputs()}
        return cls(spec, tokens, network, token_types=SEGMENT_IDS in names)

    def vectors(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return one unit vector per text, in the order given."""
        out = np.zeros((len(texts), self.spec.dimensions), np.float32)
        ids = [self._tokens.ids(text, self.spec.max_tokens) for text in texts]
        order = sorted(range(len(texts)), key=lambda i: len(ids[i]))
        for start in range(0, len(order), BATCH):
            batch = order[start : start + BATCH]
            out[batch] = self._pooled([ids[i] for i in batch])
        return out

    async def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return one unit vector per text, computed off the event loop."""
        return await asyncio.to_thread(self.vectors, texts)

    def _pooled(self, batch: list[list[int]]) -> NDArray[np.float32]:
        width = max(len(ids) for ids in batch)
        tokens = np.full((len(batch), width), PADDING_ID, np.int64)
        mask = np.zeros((len(batch), width), np.int64)
        for row, ids in enumerate(batch):
            tokens[row, : len(ids)] = ids
            mask[row, : len(ids)] = 1
        feeds: Feeds = {"input_ids": tokens, "attention_mask": mask}
        if self._token_types:
            feeds[SEGMENT_IDS] = np.zeros_like(tokens)
        hidden = self._network(feeds)
        if self.spec.pooling is Pooling.CLS:
            pooled = hidden[:, 0]
        else:
            weights = mask[..., None].astype(np.float32)
            pooled = (hidden * weights).sum(axis=1) / weights.sum(axis=1)
        return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
