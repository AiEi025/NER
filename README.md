# 🏷️ Named Entity Recognition (NER) with BERT

A complete Named Entity Recognition (NER) pipeline built from scratch using **PyTorch** and **Hugging Face Transformers**.

This project demonstrates how to fine-tune a pretrained BERT model for token classification on the **Babelscape/WikiNEuRal** dataset.

---

# Project Overview

Named Entity Recognition (NER) is one of the core tasks in Natural Language Processing.

The objective is to identify and classify entities inside a sentence, such as:

* 👤 Person
* 🏢 Organization
* 📍 Location
* 📰 Miscellaneous

Example:

```
Input:
Barack Obama visited Microsoft in Seattle.

Prediction:
Barack Obama  -> PER
Microsoft     -> ORG
Seattle        -> LOC
```

---

# Dataset

Dataset:

**Babelscape/WikiNEuRal**

Language:

* English

Entity Labels:

* O
* B-PER
* I-PER
* B-ORG
* I-ORG
* B-LOC
* I-LOC
* B-MISC
* I-MISC

---

# Project Pipeline

## 1. Dataset Exploration

* Dataset inspection
* Entity distribution analysis
* Label frequency visualization
* Sentence length analysis
* Maximum sequence length selection

---

## 2. Tokenization

Tokenizer:

```
bert-base-cased
```

Techniques used:

* WordPiece Tokenization
* Dynamic Padding
* Truncation
* Attention Mask
* Special Tokens

---

## 3. Label Alignment

Since BERT splits words into subwords, token labels must be aligned correctly.

Implemented:

* `word_ids()`
* Label alignment
* Ignore index (`-100`) for special tokens and subword tokens

---

## 4. Data Pipeline

Implemented custom preprocessing using:

* Dataset.map()
* DataCollatorForTokenClassification
* DataLoader
* Custom batching

---

## 5. Model

Model:

```
AutoModelForTokenClassification
```

Pretrained checkpoint:

```
bert-base-cased
```

Output:

```
(batch_size,
 sequence_length,
 num_labels)
```

---

## 6. Training

Framework:

* PyTorch

Optimizer:

* AdamW

Loss:

* CrossEntropyLoss (handled internally by Hugging Face)

Training Steps:

* Forward pass
* Backpropagation
* Optimizer step
* Gradient update

---

## 7. Evaluation

Evaluation metric:

* SeqEval

Reported Metrics:

* Precision
* Recall
* F1-score

Post-processing includes:

* Removing ignored labels (`-100`)
* Converting label IDs to entity names
* Computing entity-level metrics

---

## Results

Example metrics:

```
Validation Loss : ...
Precision       : ...
Recall          : ...
F1 Score        : ...
```

*(Replace these values with your final results after training.)*

---

# Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Evaluate
* SeqEval
* NumPy

---

# Repository Structure

```
NER-with-BERT/

│── train.ipynb
│── inference.ipynb
│── README.md
│── requirements.txt

├── figures/
│   ├── label_distribution.png
│   ├── sentence_length.png
│   └── prediction_example.png
```

---

# What I Learned

Throughout this project I learned:

* Token Classification pipeline
* WordPiece tokenization
* Label alignment
* Dynamic padding
* Data collators
* Fine-tuning pretrained BERT models
* Entity-level evaluation with SeqEval
* Building an end-to-end NER system using PyTorch and Hugging Face

---

# Future Improvements

* Train on multilingual datasets
* Experiment with RoBERTa and DeBERTa
* Hyperparameter tuning
* Learning rate scheduling
* Early stopping
* Model checkpointing
* Deploy the model with Gradio or FastAPI

---

# License

This project is released under the MIT License.
