# Transformer from Scratch — English to Norwegian Translation

This repository contains a complete implementation of the Transformer architecture from the paper [“Attention Is All You Need”](https://arxiv.org/abs/1706.03762), built entirely from scratch using PyTorch.

The model is trained on the `en-no` language pair from the `opus_books` dataset to perform English → Norwegian translation.

---

## Features

- **Tokenization** with `tokenizers` library (`WordLevel`)
- **Custom Transformer** with multi-head attention, layer norm, masking, and position-wise feedforward
- Supports `[PAD]`, `[SOS]`, `[EOS]`, `[UNK]` tokens
- **Custom Dataset** class with padding, attention masks, and label construction
- **Training loop** with:
  - Label smoothing
  - TensorBoard logging
  - Checkpointing and resume support
- **Greedy decoding** inference function to generate translations

---

## Dataset

- `Helsinki-NLP/opus_books` from Hugging Face
- Language pair: `en-no` (English → Norwegian)
- Tokenizers are trained from scratch on this dataset and saved as JSON

---

## Example

```python
sentence = "I would like to go to the library."
translation = translate_sentence(sentence, model, tokenizer_src, tokenizer_tgt, config)
print("→", translation)
