import pandas as pd
from torch.utils.data import Dataset

class English2FrenchDataset(Dataset):
    def __init__(self, csv_file, tokenizer, context_window):
        self.dataframe = pd.read_csv(csv_file, usecols=csv_columns)
        self.tokenizer = tokenizer
        self.context_window = context_window

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        source = self.dataframe.iloc[idx]['English words/sentences']
        target = self.dataframe.iloc[idx]['French words/sentences']

        source_encoded = self.tokenizer.encode_plus(
            source,
            max_length = context_window,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        target_encoded = self.tokenizer.encode_plus(
            target,
            max_length = context_window,
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

