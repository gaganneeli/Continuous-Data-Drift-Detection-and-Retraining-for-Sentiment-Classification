# Continuous Data Drift Detection and Retraining for Sentiment Classification

## Project Overview

This project implements a continuous training and deployment pipeline for a transformer-based text classification model. The main objective is to classify movie reviews as positive or negative using the BERT model. The system includes mechanisms to detect data drift, which can affect the model's performance over time, and triggers automatic retraining to maintain accuracy.

## Key Objectives

- **Sentiment Classification**: Classify movie reviews as good (positive) or bad (negative).
- **Data Drift Detection**: Monitor incoming data to identify significant changes that may impact model effectiveness.
- **Automatic Retraining**: Retrain the model automatically when performance degrades due to data drift.

## Model

The core of this project is a BERT-based model for sentiment analysis. BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art transformer model that excels in understanding context in natural language processing tasks. In this project, the model is fine-tuned on a labeled dataset of movie reviews to predict their sentiment accurately.

### Model Features:

- **Input**: Movie reviews as text.
- **Output**: Sentiment classification (positive or negative).
- **Training**: The model is trained using labeled data to learn the nuances of sentiment in reviews.

## Drift Detection

To detect data drift, we utilize the **Wasserstein Distance**, which measures the difference between the distributions of incoming data and the training data. A significant shift in the distribution indicates potential data drift, prompting the system to initiate retraining procedures.

## Technologies Used

- Python
- BERT (from Hugging Face Transformers)
- Scikit-learn
- Pandas
- NumPy

## How to Use

1. **Train the Model**: Run the training script to build the initial BERT model for sentiment analysis.
2. **Monitor for Data Drift**: Continuously monitor incoming reviews for changes that could affect model performance using Wasserstein Distance for drift score calculation.
3. **Retrain as Needed**: Automatically retrain the model when data drift is detected to ensure consistent accuracy.

## Conclusion

This project aims to ensure that the sentiment classification model remains robust and reliable in real-world applications by continuously monitoring and adapting to changing data patterns.
