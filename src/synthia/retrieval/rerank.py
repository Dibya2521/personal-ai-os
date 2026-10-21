"""Scores for a question and each passage, read together by a cross-encoder.

A search by vectors compares a question with passages each embedded on its
own, so a passage that only shares the question's topic can rank above one
that answers it. A cross-encoder reads ``[CLS] question [SEP] passage
[SEP]`` as one input and returns one number for the pair, higher for a
better answer, so it can reorder the first passages a search finds. It costs
a pass of the network per pair (about 110 ms for a 40-line passage, measured
on a busy laptop CPU), which is why it reorders only the first few.

The ONNX network runs on the CPU through onnxruntime, with half the threads,
in batches of the size the catalogue gives the model; pairs are sorted by
length so a batch pads to its longest pair. Scoring blocks, so
:meth:`CrossEncoder.rank` runs it off the event loop.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

from synthia.kernel.errors import SynthiaError
from synthia.retrieval.embed import PADDING_ID, SEGMENT_IDS, Feeds, threads
from synthia.retrieval.wordpiece import Pair, WordPiece

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from synthia.models.catalogue import Reranker

type Network = Callable[[Feeds], NDArray[np.float32]]
"""Token ids, attention mask and token types in; one row per pair out,
shaped (pairs, 1)."""


class RerankError(SynthiaError):
    """A reranker that cannot be loaded, or is not the one catalogued."""


class CrossEncoder:
    """A reranker, ready to score how well each passage answers a question."""

    def __init__(self, spec: Reranker, tokens: WordPiece, network: Network) -> None:
        """Score with ``network``, fed ``tokens``' pairs, as ``spec`` describes.

        Raises:
            RerankError: If the network does not give one number per pair.
        """
        self.spec = spec
        self._tokens = tokens
        self._network = network
        shape = self._network(_feeds([tokens.pair("a", "b", spec.max_tokens)])).shape
        if shape != (1, 1):
            message = f"{spec.id} gives {shape} numbers per pair, not (1, 1)"
            raise RerankError(message)

    @classmethod
    def load(cls, directory: Path, spec: Reranker) -> CrossEncoder:
        """Load ``spec``'s installed files from ``directory``.

        Raises:
            RerankError: If a file is missing, unreadable or not the model the
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
        # onnxruntime raises 15 classes with no base but Exception
        # (InvalidProtobuf for a damaged file, NoSuchFile, Fail...).
        except Exception as error:
            message = f"cannot load the reranker {spec.id}: {error}"
            raise RerankError(message) from error

        def network(feeds: Feeds) -> NDArray[np.float32]:
            return np.asarray(session.run(None, feeds)[0], np.float32)

        return cls(spec, tokens, network)

    def scores(self, question: str, passages: Sequence[str]) -> list[float]:
        """Return one score per passage, in the order given; higher answers better."""
        pairs = [self._tokens.pair(question, p, self.spec.max_tokens) for p in passages]
        out = np.zeros(len(pairs), np.float32)
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i].ids))
        size = self.spec.batch
        for start in range(0, len(order), size):
            batch = order[start : start + size]
            out[batch] = self._network(_feeds([pairs[i] for i in batch]))[:, 0]
        return [float(score) for score in out]

    async def rank(self, question: str, passages: Sequence[str]) -> list[float]:
        """Return :meth:`scores`, computed off the event loop."""
        return await asyncio.to_thread(self.scores, question, passages)


def _feeds(batch: Sequence[Pair]) -> Feeds:
    width = max(len(pair.ids) for pair in batch)
    tokens = np.full((len(batch), width), PADDING_ID, np.int64)
    mask = np.zeros((len(batch), width), np.int64)
    types = np.zeros((len(batch), width), np.int64)
    for row, pair in enumerate(batch):
        tokens[row, : len(pair.ids)] = pair.ids
        mask[row, : len(pair.ids)] = 1
        types[row, : len(pair.types)] = pair.types
    return {"input_ids": tokens, "attention_mask": mask, SEGMENT_IDS: types}
