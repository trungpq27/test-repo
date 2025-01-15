from typing import Optional
import yaml
from transformers import set_seed
from functools import partial
import datasets
from datasets import DatasetDict, IterableDatasetDict
import logging
import os
from torch.utils.data import IterableDataset
from karbon_asr.training.utlis import load_maybe_streaming_dataset, ViNormalizer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.trainer_pt_utils import IterableDatasetShard
from karbon_asr.training.schemas import (
    ModelArguments,
    DataTrainingArguments,
    DataCollatorSpeechSeq2SeqWithPadding
)

import evaluate

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)

logger = logging.getLogger(__name__)

class WhisperTrainer:
    
    def __init__(
        self, 
        model_args: Optional[ModelArguments] = None, 
        data_args: Optional[DataTrainingArguments] = None, 
        training_args: Optional[Seq2SeqTrainingArguments] = None, 
        config_path: Optional[str] = None
    ):
        
        if config_path:
            # Đọc file YAML và ánh xạ vào các dataclass
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            self.model_args = ModelArguments(**config.get("ModelArguments", {}))
            self.data_args = DataTrainingArguments(**config.get("DataTrainingArguments", {}))
            self.training_args = Seq2SeqTrainingArguments(**config.get("Seq2SeqTrainingArguments", {}))
        else:
            # Sử dụng trực tiếp các dataclass đã truyền vào
            if not (model_args and data_args and training_args):
                raise ValueError("You must provide either all dataclass arguments or a valid config_path.")
            
            self.model_args = model_args
            self.data_args = data_args
            self.training_args = training_args
            
        self.raw_datasets_features = None
        self.last_checkpoint = None
        
        if (
            os.path.isdir(self.training_args.output_dir)
            and self.training_args.do_train
            and not self.training_args.overwrite_output_dir
        ):
            self.last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
            if self.last_checkpoint is None and len(os.listdir(self.training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({self.training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif (
                self.last_checkpoint is not None and self.training_args.resume_from_checkpoint is None
            ):
                logger.info(
                    f"Checkpoint detected, resuming training at {self.last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        set_seed(self.training_args.seed)
        
        
        ##### Config, Feature Extractor, Tokenizer and model init ####
        self.config = AutoConfig.from_pretrained(
            (
                self.model_args.config_name
                if self.model_args.config_name
                else self.model_args.model_name_or_path
            ),
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )

        self.config.update(
            {
                "forced_decoder_ids": self.model_args.forced_decoder_ids,
                "suppress_tokens": self.model_args.suppress_tokens,
            }
        )

        if self.training_args.gradient_checkpointing:
            self.config.update({"use_cache": False})

        # SpecAugment for whisper models
        if getattr(self.config, "model_type", None) == "whisper":
            self.config.update({"apply_spec_augment": self.model_args.apply_spec_augment})

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            (
                self.model_args.feature_extractor_name
                if self.model_args.feature_extractor_name
                else self.model_args.model_name_or_path
            ),
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            (
                self.model_args.tokenizer_name
                if self.model_args.tokenizer_name
                else self.model_args.model_name_or_path
            ),
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_args.model_name_or_path,
            config=self.config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        
        if self.model.config.decoder_start_token_id is None:
            raise ValueError(
                "Make sure that `config.decoder_start_token_id` is correctly defined"
            )

        if self.model_args.freeze_feature_encoder:
            self.model.freeze_feature_encoder()

        if self.model_args.freeze_encoder:
            self.model.freeze_encoder()
            self.model.model.encoder.gradient_checkpointing = False

        # set language and task for generation and re-enable cache
        self.model.generate = partial(
            self.model.generate,
            language=self.data_args.language,
            task=self.data_args.task,
            use_cache=True,
        )

        if self.data_args.language is not None:
            # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
            self.tokenizer.set_prefix_tokens(language=self.data_args.language, task=self.data_args.task)
        
        
        
    def _load_dataset(self):
        raw_datasets = IterableDatasetDict() if self.data_args.streaming else DatasetDict()

        if self.training_args.do_train:
            raw_datasets["train"] = load_maybe_streaming_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
                split=self.data_args.train_split_name,
                streaming=self.data_args.streaming,
                token=self.model_args.token,
            )

        if self.training_args.do_eval:
            raw_datasets["eval"] = load_maybe_streaming_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
                split=self.data_args.eval_split_name,
                streaming=self.data_args.streaming,
                token=self.model_args.token,
            )

        self.raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys())

        if self.data_args.audio_column_name not in self.raw_datasets_features:
            raise ValueError(
                f"--audio_column_name '{self.data_args.audio_column_name}' not found in dataset '{self.data_args.dataset_name}'. "
                "Make sure to set `--audio_column_name` to the correct audio column - one of "
                f"{', '.join(self.raw_datasets_features)}."
            )

        if self.data_args.text_column_name not in self.raw_datasets_features:
            raise ValueError(
                f"--text_column_name {self.data_args.text_column_name} not found in dataset '{self.data_args.dataset_name}'. "
                "Make sure to set `--text_column_name` to the correct text column - one of "
                f"{', '.join(self.raw_datasets_features)}."
            )
            
        # Resample speech dataset if necessary
        dataset_sampling_rate = (
        next(iter(raw_datasets.values()))
        .features[self.data_args.audio_column_name]
        .sampling_rate
        )
        if dataset_sampling_rate != self.feature_extractor.sampling_rate:
            raw_datasets = raw_datasets.cast_column(
                self.data_args.audio_column_name,
                datasets.features.Audio(sampling_rate=self.feature_extractor.sampling_rate),
            )
            
        return raw_datasets
    
    
    def _vectorize_and_collate(self, raw_datasets):
        max_input_length = (
            self.data_args.max_duration_in_seconds * self.feature_extractor.sampling_rate
        )
        min_input_length = (
            self.data_args.min_duration_in_seconds * self.feature_extractor.sampling_rate
        )
        audio_column_name = self.data_args.audio_column_name
        text_column_name = self.data_args.text_column_name
        model_input_name = self.feature_extractor.model_input_names[0]
        do_lower_case = self.data_args.do_lower_case
        do_remove_punctuation = self.data_args.do_remove_punctuation

        forward_attention_mask = (
            getattr(self.config, "model_type", None) == "whisper"
            and getattr(self.config, "apply_spec_augment", False)
            and getattr(self.config, "mask_time_prob", 0) > 0
        )

        if self.data_args.max_train_samples is not None:
            raw_datasets["train"] = (
                raw_datasets["train"].take(self.data_args.max_train_samples)
                if self.data_args.streaming
                else raw_datasets["train"].select(range(self.data_args.max_train_samples))
            )

        if self.data_args.max_eval_samples is not None:
            raw_datasets["eval"] = (
                raw_datasets["eval"].take(self.data_args.max_eval_samples)
                if self.data_args.streaming
                else raw_datasets["eval"].select(range(self.data_args.max_eval_samples))
            )

        
        def prepare_dataset(batch):
            # process audio
            sample = batch[audio_column_name]
            inputs = self.feature_extractor(
                sample["array"],
                sampling_rate=sample["sampling_rate"],
                return_attention_mask=forward_attention_mask,
            )
            # process audio length
            batch[model_input_name] = inputs.get(model_input_name)[0]
            batch["input_length"] = len(sample["array"])
            if forward_attention_mask:
                batch["attention_mask"] = inputs.get("attention_mask")[0]

            # process targets
            input_str = (
                batch[text_column_name].lower()
                if do_lower_case
                else batch[text_column_name]
            )
            if self.data_args.language == "vi":
                normalizer = ViNormalizer()
            else:
                normalizer = BasicTextNormalizer()
                
            input_str = normalizer(input_str).strip()
            batch["labels"] = self.tokenizer(input_str).input_ids
            # compute labels length **with** special tokens! -> total label length
            batch["labels_length"] = len(batch["labels"])

            return batch

        with self.training_args.main_process_first(desc="dataset map pre-processing"):
            vectorized_datasets = raw_datasets.map(
                prepare_dataset,
                remove_columns=self.raw_datasets_features,
            ).with_format("torch")

            if self.training_args.do_train and self.data_args.streaming:
                # manually shuffle if streaming (done by the trainer for non-streaming)
                vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(
                    buffer_size=self.data_args.shuffle_buffer_size,
                    seed=self.training_args.seed,
                )

        # filter training data that is shorter than min_input_length or longer than
        # max_input_length
        def is_audio_in_length_range(length):
            return min_input_length < length < max_input_length

        max_label_length = self.model.config.max_length

        def filter_labels(labels_length):
            """Filter label sequences longer than max length (448)"""
            return labels_length < max_label_length

        if self.training_args.do_train:
            vectorized_datasets["train"] = vectorized_datasets["train"].filter(
                is_audio_in_length_range,
                input_columns=["input_length"],
            )
        vectorized_datasets = vectorized_datasets.filter(
            filter_labels, input_columns=["labels_length"]
        )
        
        # Create a single speech processor
        # make sure all processes wait until data is saved
        with self.training_args.main_process_first():
            # only the main process saves them
            if is_main_process(self.training_args.local_rank):
                # save feature extractor, tokenizer and config
                self.feature_extractor.save_pretrained(self.training_args.output_dir)
                self.tokenizer.save_pretrained(self.training_args.output_dir)
                self.config.save_pretrained(self.training_args.output_dir)

        processor = AutoProcessor.from_pretrained(self.training_args.output_dir)

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
            forward_attention_mask=forward_attention_mask,
        )
        
        return vectorized_datasets, data_collator
        
        
    def train(self):
        raw_datasets = self._load_dataset()
        vectorized_datasets, data_collator = self._vectorize_and_collate(raw_datasets)
        
        wer_metric = evaluate.load("wer")
        chrf_metric = evaluate.load("chrf")
        sacrebleu = evaluate.load("sacrebleu")
        do_normalize_eval = self.data_args.do_normalize_eval

        def compute_metrics(pred):
            pred_ids = pred.predictions

            pred.label_ids[pred.label_ids == -100] = self.tokenizer.pad_token_id

            pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            # we do not want to group tokens when computing the metrics
            label_str = self.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

            if do_normalize_eval:
                if self.data_args.language == "vi":
                    normalizer = ViNormalizer()
                else:
                    normalizer = BasicTextNormalizer()
                
                pred_str = [normalizer(pred) for pred in pred_str]
                label_str = [normalizer(label) for label in label_str]
                # filtering step to only evaluate the samples that correspond to non-zero references:
                pred_str = [
                    pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0
                ]
                label_str = [
                    label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0
                ]

            if self.data_args.task == "transcribe":
                return {
                    "wer": 100
                    * wer_metric.compute(predictions=pred_str, references=label_str)
                }
            elif self.data_args.task == "translate":
                return {
                    "sacrebleu": sacrebleu.compute(
                        predictions=pred_str, references=label_str
                    )["score"],
                    "chrf++": chrf_metric.compute(
                        predictions=pred_str,
                        references=label_str,
                        word_order=2,
                        eps_smoothing=True,
                    )["score"],
                }
            else:
                raise ValueError(f"Unexpected task={self.data_args.task}")

        class ShuffleCallback(TrainerCallback):
            def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
                if isinstance(train_dataloader.dataset, IterableDatasetShard):
                    pass  # set_epoch() is handled by the Trainer
                elif isinstance(train_dataloader.dataset, IterableDataset):
                    train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)

        # Initialize Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=vectorized_datasets["train"] if self.training_args.do_train else None,
            eval_dataset=vectorized_datasets["eval"] if self.training_args.do_eval else None,
            processing_class=self.feature_extractor,
            data_collator=data_collator,
            compute_metrics=(
                compute_metrics if self.training_args.predict_with_generate else None
            ),
            callbacks=[ShuffleCallback()] if self.data_args.streaming else None,
        )

        # 12. Training
        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif self.last_checkpoint is not None:
                checkpoint = self.last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the feature extractor too for easy upload

            metrics = train_result.metrics
            if self.data_args.max_train_samples:
                metrics["train_samples"] = self.data_args.max_train_samples
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            
        kwargs = {
            "finetuned_from": self.model_args.model_name_or_path,
            "tasks": "automatic-speech-recognition",
        }
        if self.data_args.dataset_name is not None:
            kwargs["dataset_tags"] = self.data_args.dataset_name
            if self.data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = self.data_args.dataset_config_name
                kwargs["dataset"] = (
                    f"{self.data_args.dataset_name} {self.data_args.dataset_config_name}"
                )
            else:
                kwargs["dataset"] = self.data_args.dataset_name
            if "common_voice" in self.data_args.dataset_name:
                kwargs["language"] = self.data_args.dataset_config_name.split("-")[0]
            if self.model_args.model_index_name is not None:
                kwargs["model_name"] = self.model_args.model_index_name

        if self.training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)