# FAQ Chatbot with Transformer Models

A question-answering chatbot that uses pre-trained transformer models to extract answers from a given context — built and compared across BERT, RoBERTa, and DistilBERT.

## 📌 Project Overview

This project implements an **extractive question-answering FAQ chatbot** using three state-of-the-art transformer models from Hugging Face. Given a passage of text as context (a fictional car dealership — Sunset Motors), each model is tasked with finding the most relevant span of text that answers a user's question.

The project serves as a practical comparison of how different transformer architectures handle the same QA task, with differences in accuracy, verbosity, and token handling.

## 🛠️ Libraries Used

- **Deep Learning / NLP:** `transformers`, `torch`
- **Models:** `BERT`, `RoBERTa`, `DistilBERT` (via Hugging Face)

## 🤖 Models Compared

| Model | Hugging Face Checkpoint |
|---|---|
| BERT | `bert-large-uncased-whole-word-masking-finetuned-squad` |
| RoBERTa | `deepset/roberta-base-squad2` |
| DistilBERT | `distilbert-base-uncased-distilled-squad` |

All models are fine-tuned on **SQuAD** (Stanford Question Answering Dataset), a benchmark reading comprehension dataset.

## 🚀 Key Implementation Steps

**1. Context & Question Input**
- A fixed context paragraph about Sunset Motors is used as the knowledge base.
- User questions are encoded alongside the context and passed into each model.

**2. Tokenization & Segment Embeddings**
- Each model uses its own tokenizer to encode the `[question, context]` pair.
- For BERT, segment IDs (0 for question, 1 for context) are manually constructed to separate the two inputs.

**3. Answer Span Prediction**
- Each model outputs `start_logits` and `end_logits` over the token sequence.
- `argmax` is applied to identify the most likely start and end token positions of the answer.

**4. Post-Processing**
- BERT requires manual subword token reconstruction (handling `##` prefixes).
- RoBERTa and DistilBERT use `convert_tokens_to_string()` for cleaner decoding.

## 📊 Sample Results

| Question | BERT | RoBERTa | DistilBERT |
|---|---|---|---|
| Where is the dealership located? | `crestwood` | `Crestwood` | `crestwood` |
| What make of cars are available? | `ford, toyota, honda, chevrolet, and bmw` | `new and pre-owned cars, trucks, SUVs...` | `used cars` |
| How large is the dealership? | `10 acres` | `over 10 acres` | `10 acres` |

**Key Observations:**
- **RoBERTa** tends to return the most natural, complete answer spans.
- **BERT** returns precise but lowercase answers due to its uncased tokenizer; requires manual subword stitching.
- **DistilBERT** is the lightest model but can return less precise answers (e.g., "used cars" instead of listing all brands).

## 📂 Repository Structure

```
├── FAQ_Chatbot.ipynb     # Main notebook with all three model implementations
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 👩‍💻 How to Run

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Open the notebook in Jupyter
4. Run all cells from top to bottom

> **Note:** The first run will automatically download the pre-trained model weights from Hugging Face (~1–1.3 GB total for all three models). An internet connection is required.

---
**Developed by:**   
Nicole Kaye A. Cardel    
nkcardel@gmail.com  
*Software Designer & Developer*
