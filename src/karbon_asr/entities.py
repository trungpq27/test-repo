from __future__ import annotations

import enum
from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

import numpy as np


class BackEndTranscriber(enum.Enum):
    faster_whisper = "faster-whisper"
    transformers = "transformers"


@dataclass
class AudioRaw:
    array: np.ndarray
    sampling_rate: int

    def to_hf_input(self) -> dict[str, int | np.ndarray]:
        return {"sampling_rate": self.sampling_rate, "raw": self.array}


@dataclass
class AudioFile:
    """Class to represent the input Audio file to the ASR pipeline.

    Raises:
        AssertionError: If no input or more than one input are provided.

    Returns:
        str or bytes or np.ndarray: either a path or raw bytes or numpy array.
    """

    path: str = None
    bytes: bytes = None
    binary: BinaryIO = None
    array: np.ndarray = None
    raw: AudioRaw = None

    def __post_init__(self):
        """Post initialization operation.

        Raises:
            AssertionError: If no input or more than one input are provided.
        """
        if (
            sum(
                1
                for x in (self.path, self.bytes, self.binary, self.array, self.raw)
                if x is not None
            )
            != 1
        ):
            raise ValueError(
                "Exactly ONE input: a path (local or url), binary or an array must be provided."
            )

    def to_transformers_input(
        self,
    ) -> str | bytes | np.ndarray | dict[str, int | np.ndarray]:
        """Return either path, binary or array based on input.

        Returns:
            str or bytes or np.ndarray: either a path or raw bytes or numpy array.
        """
        if self.path:
            return self.path
        elif self.bytes:
            return self.bytes
        elif self.binary:
            return self.binary.read()
        elif self.array is not None:
            return self.array
        else:
            return self.raw.to_hf_input()

    def to_faster_whisper_input(self) -> str | BinaryIO | np.ndarray:
        if self.path:
            return self.path
        elif self.bytes:
            return BytesIO(self.bytes)
        elif self.binary:
            return self.binary
        elif self.array is not None:
            return self.array
        else:
            raise NotImplementedError(
                "Faster whisper input now doesn't support AudioRaw."
            )


@dataclass
class Chunk:
    """Class to represent a single chunk in the ASR pipeline's output with return_timestamps set to True.

    Attributes:
        start_time (float): The start time of the chunk in seconds.
        end_time (float): The end time of the chunk in seconds.
        text (str): The text content of the chunk.
    """

    start_time: float
    end_time: float
    text: str
    avg_logprob: float = None


@dataclass
class Transcription:
    """Class to represent the output of the ASR pipeline with return_timestamps set to True.

    Attributes:
        full_text (str): The full transcription of the audio
        chunks (list[Chunk]): List of chunks in the transcription if return_timestamps set to True.
    """

    full_text: str
    chunks: list[Chunk] | None = None

    @classmethod
    def from_transformer_output(cls, result: dict) -> Transcription:
        chunks = None

        if "chunks" in result:
            chunks = [
                Chunk(
                    start_time=chunk["timestamp"][0],
                    end_time=chunk["timestamp"][1],
                    text=chunk["text"],
                )
                for chunk in result["chunks"]
            ]
        return Transcription(full_text=result["text"], chunks=chunks)
