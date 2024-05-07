## Usage

In this repository, we provide the translator model for English to German.

The .ipynb file includes the code for model training and inference. You might download it and run on your device. It would take you 6-7 hours to train the model with our provided 15.9MB data on T4 GPU on Google Colab.

## Model
The translator mdoel is built upon model T5. T5 (Text-to-Text Transfer Transformer) model is introduced by Google in 2019. It is designed with a unified approach to handle a variety of NLP tasks by converting every task into a text-to-text format. 

T5 is built on the Transformer model, employing a sequence-to-sequence (seq2seq) architecture with both an encoder and a decoder. Its primary training objective involves a denoising task where the model learns to reconstruct the original text from corrupted input.It has 223 million of parameters while models such as GPT4 has 300 billion parameters. 

Its training utilizes a teacher forcing strategy, where during training, the model is shown the entire correct output sequence at once, allowing it to learn more efficiently. This is in contrast to typical training where only the next token is predicted. The model is pre-trained on a mixture of unsupervised and supervised tasks converted into a text-to-text format.

Due to its seq2seq framework, T5 excels in tasks that involve transformations of text, such as translation, summarization, question answering, and even tasks that can be reframed as text generation, like classification (by generating labels as text).

However,  it performs worse in handling specific knowledge queries nor in a "few-shot" or "zero-shot" learning mode, where it should generate responses based on a small number of examples or no specific task-focused training.

