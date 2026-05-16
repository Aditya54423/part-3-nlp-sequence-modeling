# Part 3: NLP and Sequence Modeling Mini Project

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow) 
![NLP](https://img.shields.io/badge/NLP-Sequence_Modeling-green)

##  Project Overview
This project compares traditional machine learning approaches with sequence-based deep learning for text classification. The goal is to understand how raw text is transformed into numerical vectors and how models like LSTMs capture the order and context of words.

## Task 1 & 2: Dataset & Preprocessing
Analysis: Evaluated text record counts, class distribution, and average sentence length.

Cleaning: Implemented a pipeline including lowercasing, removal of special characters/symbols, and tokenization.

Sequence Handling: For the deep learning model, sequences were padded to a uniform length to ensure consistent input tensor shapes.

## Task 3: Text Vectorization
The project implements three distinct vectorization strategies:

TF-IDF: For the baseline models (captures word importance but loses order).

Tokenization: Converting words to unique integer indices.

Embeddings: Dense vectors that map words into a continuous vector space where similar words are closer together.

## Task 4 & 5: Model Comparison
I compared traditional baselines against a sequence model:

Baselines: Multinomial Naive Bayes and Logistic Regression using TF-IDF.

Sequence Model: A Bidirectional LSTM (BiLSTM) architecture.

Architecture: Embedding Layer → BiLSTM Layer → Dense (ReLU) → Output (Softmax).

Why BiLSTM? It processes text in both forward and backward directions, capturing context from both the start and end of a sentence.

## Task 6: NLP Concepts & Reflections
RNN Limitations: Standard RNNs suffer from "Vanishing Gradients," making them forget information from the beginning of long sentences.

LSTM Solution: LSTMs use "Gates" (Input, Forget, Output) to regulate information flow, allowing them to maintain long-term dependencies.

Attention & Transformers: Attention mechanisms allow models to focus on specific relevant words regardless of distance. Transformers use "Self-Attention" to process
entire sentences in parallel, making them the backbone of modern LLMs.

