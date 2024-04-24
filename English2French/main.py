import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer

# Constants
batch_size=2

class English2FrenchDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        # Read the CSV file directly within the class constructor
        self.dataframe = pd.read_csv(
            csv_file,
            usecols=['English words/sentences', 'French words/sentences']
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        source = self.dataframe.iloc[idx]['English words/sentences']
        target = self.dataframe.iloc[idx]['French words/sentences']

        # Encode the source and target text
        source_encoded = self.tokenizer.encode_plus(
            source,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target_encoded = self.tokenizer.encode_plus(
            target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'source_text': source,
            'source_input_ids': source_encoded['input_ids'].flatten(),
            'target_text': target,
            'target_input_ids': target_encoded['input_ids'].flatten()
        }

def main():
  # Tokenization
  tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base');
  # Data Preparation
  dataset = English2FrenchDataset('eng_french.csv', tokenizer)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  # Testing
  iterator = iter(dataset)
  print(next(iterator)['source_text'])
  print(next(iterator)['target_text'])
  print(next(iterator)['source_input_ids'][0:10])
  print(next(iterator)['target_input_ids'][0:10])


if __name__ == "__main__":
    main()

