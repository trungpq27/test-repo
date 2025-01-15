import re
from datasets import interleave_datasets, load_dataset

def load_maybe_streaming_dataset(
    dataset_name, dataset_config_name, split="train", streaming=True, **kwargs
):
    """
    Utility function to load a dataset in streaming mode. For datasets with multiple splits,
    each split is loaded individually and then splits combined by taking alternating examples from
    each (interleaving).
    """
    if "+" in split:
        # load multiple splits separated by the `+` symbol with streaming mode
        dataset_splits = [
            load_dataset(
                dataset_name,
                dataset_config_name,
                split=split_name,
                streaming=streaming,
                **kwargs,
            )
            for split_name in split.split("+")
        ]
        # interleave multiple splits to form one dataset
        interleaved_dataset = interleave_datasets(dataset_splits)
        return interleaved_dataset
    else:
        # load a single split *with* streaming mode
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            split=split,
            streaming=streaming,
            **kwargs,
        )
        return dataset
    
    
class ViNormalizer():
    def __call__ (self, text):
        text = re.sub(r'[^\w\sàáảãạăắặẳẵâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựýỷỹỵđ]', '', text)
        text = ' '.join([word.lower().strip() for word in text.split()])
        return text
