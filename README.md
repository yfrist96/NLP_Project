# Sentiment Analysis with NLP Models

# Project Overview
This project implements and compares multiple machine learning models for sentiment analysis on the Stanford Sentiment Treebank dataset. The goal is to classify sentences from movie reviews as either positive or negative based on their sentiment.

I've implemented and evaluated four different models:
1. Log-Linear Model with One-Hot Encoding
2. Log-Linear Model with Word2Vec Embeddings
3. LSTM Model with Word2Vec Embeddings
4. Transformer Model

# Results & Comparisons
![image](https://github.com/user-attachments/assets/ee48636b-2ff7-4981-ba99-29f9789c5bd7)

# Key Insights
1. One-Hot Encoding Performs Poorly - Since one-hot encoding doesn’t capture word relationships, it fails on unseen words and negations.
2. Word2Vec Improves Performance - Pre-trained embeddings allow the model to generalize better by capturing word meanings.
3. LSTM Handles Negation & Sequential Context Better - Captures dependencies across words, leading to better negation handling.
4. Transformers Achieve the Best Accuracy - Handles long-range dependencies and understands context more effectively than RNN-based models.
   
# Installation and Setup

