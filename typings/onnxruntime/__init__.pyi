# The part of onnxruntime SYNTHIA uses: the package ships without type hints.

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

class SessionOptions:
    intra_op_num_threads: int

class NodeArg:
    @property
    def name(self) -> str: ...

class InferenceSession:
    def __init__(
        self,
        path_or_bytes: str,
        sess_options: SessionOptions | None = None,
        providers: Sequence[str] | None = None,
    ) -> None: ...
    def get_inputs(self) -> list[NodeArg]: ...
    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, NDArray[np.int64]],
    ) -> list[NDArray[np.float32]]: ...
