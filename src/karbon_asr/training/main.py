import os
import sys
import warnings

from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
)

from karbon_asr.training.schemas import (
    ModelArguments,
    DataTrainingArguments
)
from karbon_asr.training.whisper_trainer import WhisperTrainer

def main():
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("model_args", vars(model_args))
    print("data_args", vars(data_args))
    print("training_args", vars(training_args))

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        model_args.token = model_args.use_auth_token

    trainer = WhisperTrainer(model_args, data_args, training_args)
    trainer.train()

if __name__ == "__main__":
    main()
