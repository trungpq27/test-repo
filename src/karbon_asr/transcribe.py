from __future__ import annotations

import logging
import os

from karbon_asr.entities import AudioFile, BackEndTranscriber, Chunk, Transcription

LOGGER = logging.getLogger(__name__)

DEFAULT_TRANSFORMER_MODEL_ID = "karbon-ai/whisper-medium-s2tt-twi2engnew-v8.mix-v1-11gb"
DEFAULT_FASTER_WHISPER_MODEL_ID = f"{DEFAULT_TRANSFORMER_MODEL_ID}-int8"

FASTER_WHISPER_TARGET_LANGUAGE = os.environ.get(
    "KARBON_ASR_FASTER_WHISPER_TARGET_LANGUAGE", "en"
)
TRANSFORMER_TARGET_LANGUAGE = os.environ.get(
    "KARBON_ASR_TRANSFORMER_TARGET_LANGUAGE", "english"
)
TASK_TYPE = os.environ.get("KARBON_ASR_TASK_TYPE", "translate")


class ASRTranscriber:
    def __init__(
        self,
        backend: BackEndTranscriber = BackEndTranscriber.faster_whisper,
        model_id=None,
        target_language=None,
        task_type=None,
        device: int | str = "cpu",
        **kwargs,
    ):
        f"""Construct the ASRTranscriber.

        Args:
            backend (BackEndTranscriber): the backend to use for inference.
                Defaults to BackEndTranscriber.faster_whisper.
            model_id (str): Your preferred Hugging Face speech to text translation model id.
                Defaults to {DEFAULT_FASTER_WHISPER_MODEL_ID} if backend='faster_whisper'
                or to {DEFAULT_TRANSFORMER_MODEL_ID} if backend='transformers'.
            device (`int` or `str`): Defines the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`,
                or a GPU ordinal rank like `1`) on which this pipeline will be allocated.
            kwargs: Additional keyword arguments to pass to the init model function.
        """
        self._backend = backend
        self._model_id = model_id or os.environ.get("KARBON_ASR_MODEL_ID")
        self._task_type = TASK_TYPE
        if self._model_id is None:
            self._model_id = (
                DEFAULT_FASTER_WHISPER_MODEL_ID
                if self._backend == BackEndTranscriber.faster_whisper
                else DEFAULT_TRANSFORMER_MODEL_ID
            )
            LOGGER.warning(f"Using default model_id: {self._model_id}.")

        if self._backend == BackEndTranscriber.faster_whisper:
            self._device, device_index = self.parse_device_for_faster_whisper(device)
            self._model_kwargs = {
                "compute_type": "int8",
                "cpu_threads": 0,
                "num_workers": 1,
            }
            self._model_kwargs.update(kwargs)
            logging.info(
                f"Create faster whisper model with model_id: {self._model_id}, "
                f"device_id: {device}, device_index:{device_index}, compute_type={self._model_kwargs['compute_type']}, "
                f"cpu_threads={self._model_kwargs['cpu_threads']}, num_workers={self._model_kwargs['num_workers']}."
            )
            try:
                from faster_whisper import WhisperModel
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Just install karbon-asr[faster-whisper] before using backend='faster-whisper'."
                )
            self._faster_whisper_model = WhisperModel(
                model_size_or_path=self._model_id,
                device=self._device,
                device_index=device_index,
                **self._model_kwargs,
            )
            self._target_language = (
                target_language
                if target_language is not None
                else FASTER_WHISPER_TARGET_LANGUAGE
            )
        else:
            self._device = device
            logging.info(
                f"Create faster whisper model with model_id: {self._model_id}, "
                f"device_id: {device}."
            )
            try:
                from transformers import pipeline
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Just install karbon-asr[transformers] before using backend='transformers'."
                )
            self._transformers_model = pipeline(
                "automatic-speech-recognition",
                model=self._model_id,
                device=self._device,
                **kwargs,
            )
            self._target_language = (
                target_language
                if target_language is not None
                else TRANSFORMER_TARGET_LANGUAGE
            )

    @staticmethod
    def parse_device_for_faster_whisper(device) -> tuple[str, int | list[int]]:
        if isinstance(device, str):
            assert device in ["cpu", "auto"] or device.startswith(
                "cuda:"
            ), "Device must be either 'cpu' or 'auto' or 'cuda'."
            if device.startswith("cuda:"):
                return "cuda", int(device.split(":")[1])
            else:
                return device, 0
        elif isinstance(device, int):
            if device == -1:
                return "cpu", 0
            elif device >= 0:
                return "cuda", device
            else:
                raise ValueError(
                    f"Unsupported device: {device}. Device must greater or equal to -1."
                )
        else:
            raise ValueError(
                f"Unsupported device: {device}. Device must to be either a string or an integer."
            )

    def transcribe(
        self,
        audio_file: AudioFile,
        **kwargs,
    ) -> Transcription:
        """Run the pipeline and return a Transcription object

        Args:
            audio_file (AudioFile): Path (local or url) to the audio file or the raw content (in bytes)
                of the audio file.
            kwargs: Additional keyword arguments to pass to the transcribe function.
        Returns:
            Transcription: An object to represent the pipeline output, can be extracted into full text
                or chunks with timestamps
        """
        if self._backend == BackEndTranscriber.faster_whisper:
            transcribe_kwargs = {
                "beam_size": 1,
                "language": self._target_language,
                "task": self._task_type,
            }
            transcribe_kwargs.update(kwargs)
            segments, info = self._faster_whisper_model.transcribe(
                audio_file.to_faster_whisper_input(), **transcribe_kwargs
            )
            texts = []
            list_chunks = []
            for segment in segments:
                texts.append(segment.text)
                list_chunks.append(
                    Chunk(
                        text=segment.text,
                        start_time=segment.start,
                        end_time=segment.end,
                        avg_logprob=segment.avg_logprob,
                    )
                )

            full_text = " ".join(texts)
            return Transcription(full_text=full_text, chunks=list_chunks)
        else:
            transcribe_kwargs = {
                "return_timestamps": True,
                "generate_kwargs": {
                    "task": self._task_type,
                    "language": self._target_language,
                },
                "chunk_length_s": 30,
            }
            transcribe_kwargs.update(kwargs)

            eng_transcription = self._transformers_model(
                audio_file.to_transformers_input(),
                **transcribe_kwargs,
            )
            return Transcription.from_transformer_output(eng_transcription)
