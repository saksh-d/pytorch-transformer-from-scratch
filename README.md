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

## Instructions

1. Clone the repository
```bash
git clone https://github.com/yourusername/transformer-en-no.git
cd transformer-en-no
```

2. It is recommended to use a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. Open the jupyter notebook and run all cells

```bash
jupyter notebook transformer_from_scratch.ipynb
```

This will:
- Download and preprocess the English–Norwegian dataset
- Train a custom tokenizer
- Build and train a Transformer model from scratch
- Save model checkpoints to /weights

---

## Example

```python
sentence = "I would like to go to the library."
translation = translate_sentence(sentence, model, tokenizer_src, tokenizer_tgt, config)
print("→", translation)
