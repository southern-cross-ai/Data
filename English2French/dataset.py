from torch.utils.data import Dataset
import pandas as pd

class TranslationDataset(Dataset):
    def __init__(self, csv_file, source_column, target_column, tokenizer, context_window):
        self.dataframe = pd.read_csv(csv_file, usecols=[source_column, target_column])
        self.source_column = source_column
        self.target_column = target_column
        self.tokenizer = tokenizer
        self.context_window = context_window

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        source = self.dataframe.iloc[idx][self.source_column]
        target = self.dataframe.iloc[idx][self.target_column]

        source_encoded = self.tokenizer.encode_plus(
            source,
            max_length=self.context_window,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        target_encoded = self.tokenizer.encode_plus(
            target,
            max_length=self.context_window,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Generate masks for source and target
        source_mask = (source_encoded['input_ids'].squeeze() != self.tokenizer.pad_token_id)
        target_mask = (target_encoded['input_ids'].squeeze() != self.tokenizer.pad_token_id)

        return {
            'source_text': source,
            'source_input_ids': source_encoded['input_ids'].squeeze(),
            'source_mask': source_mask,
            'target_text': target,
            'target_input_ids': target_encoded['input_ids'].squeeze(),
            'target_mask': target_mask
        }
