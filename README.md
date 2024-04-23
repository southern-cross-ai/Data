# SouthernCrossAI

<a href="https://southern-cross-ai.github.io/TranslationAI/" target="_blank">Welcome to SouthernCrossAI</a>

In this repository, we are trialing some of the earlier implementations of Seq2Seq models. Our main focus is on the Transformer architecture, not the RNN types. The two are often confused when searching online for help.

The main goal is to understand the complete encoder-decoder architecture in relation to NLP and to become comfortable with integrating these models into the Hugging Face repository.

We will not be building the tokenizer but instead will rely on ones that are already built.

## Here is a short list of Models that use the Encoder-Decoders architecture
* **Transformer** - The original architecture introduced by Vaswani et al., featuring both encoder and decoder blocks.
* **T5** (Text-To-Text Transfer Transformer) - A versatile model that reframes all NLP tasks as a text-to-text problem, using an encoder-decoder structure.
* **BART** (Bidirectional and Auto-Regressive Transformers) - Combines a bidirectional encoder (like BERT) and an autoregressive decoder, ideal for generative tasks like summarization.
* **MarianMT** - A model built specifically for machine translation, utilizing the standard encoder-decoder setup.
* **Seq2Seq** with Attention - Enhances the basic Seq2Seq architecture by incorporating attention mechanisms, typically used in machine translation.
* **UNILM** (Unified Language Model) - Uses an encoder-decoder architecture where some layers are shared, suitable for both generation and understanding tasks.
* **mT5** (Multilingual T5) - A multilingual extension of the T5 model that uses an encoder-decoder framework to handle tasks in over 100 languages.
