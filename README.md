# SouthernCrossAI

<a href="https://southern-cross-ai.github.io/TranslationAI/" target="_blank">Welcome to SouthernCrossAI</a>

In this repository, we are trialling some of the earlier implementations of Seq2Seq models. Our main focus is on the Transformer architecture, not the RNN types. The two are often confused when searching online for help.

The main goal is to understand the complete encoder-decoder architecture in relation to NLP and to become comfortable with integrating these models into the Hugging Face repository.

## We will not build the tokenizer but instead rely on already built ones.
```python
# mBERT (Multilingual BERT)
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# XLM-RoBERTa (XLM-R)
from transformers import XLMRobertaTokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# Multilingual T5
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')

# MarianMT
from transformers import MarianTokenizer
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
```





## Here is a short list of Models and Tokenizers that use the Encoder-Decoders architecture
* **Transformer** - The original architecture introduced by Vaswani et al., featuring both encoder and decoder blocks. 
* **T5** (Text-To-Text Transfer Transformer) - A versatile model that reframes all NLP tasks as a text-to-text problem, using an encoder-decoder structure. 
* **BART** (Bidirectional and Auto-Regressive Transformers) - Combines a bidirectional encoder (like BERT) and an autoregressive decoder, ideal for generative tasks like summarization.
* **MarianMT** - A model built specifically for machine translation, utilizing the standard encoder-decoder setup.
* **Seq2Seq** with Attention - Enhances the basic Seq2Seq architecture by incorporating attention mechanisms, typically used in machine translation.
* **UNILM** (Unified Language Model) - Uses an encoder-decoder architecture where some layers are shared, suitable for both generation and understanding tasks.
* **mT5** (Multilingual T5) - A multilingual extension of the T5 model that uses an encoder-decoder framework to handle tasks in over 100 languages. 
