# Movie Review Sentiment Classification

This project focuses on building different models to classify movie reviews as positive or negative using the IMDB movie review dataset. The models include a shallow neural network, LSTM, Bi-LSTM, and GRU. The goal is to evaluate the performance of each model and improve accuracy through different techniques.

## Objectives
1. Clean the dataset.
2. Split the dataset into 80% training and 20% testing sets.
3. Apply GloVe Embedding on the movie reviews.
4. Build and evaluate multiple models: Shallow Model, LSTM, Bi-LSTM, and GRU.

## Dataset
The dataset used for this project is the IMDB movie review dataset containing 50,000 reviews, evenly distributed between positive and negative sentiments. The total vocabulary size is 167,313 words, and the maximum sequence length is 1,437 words.

## Data Preprocessing
- Removed punctuation marks and HTML tags.
- Removed stop words using NLTK’s English stopword list.
- Tokenized the dataset and applied 100-dimensional GloVe embeddings for word representation.

## Models

### 1. Shallow Model
A simple model with:
- Embedding layer.
- Dense layer with 10 units (tanh activation).
- Final Dense layer with 1 unit (sigmoid activation).

### 2. LSTM
The LSTM model consists of:
- Embedding layer initialized with pre-trained GloVe vectors.
- LSTM layer with 64 units and tanh activation.
- Final Dense layer with 1 unit (sigmoid activation).

### 3. Bi-LSTM
The Bi-LSTM model features:
- Embedding layer with GloVe vectors.
- Bi-LSTM layer with 64 units.
- Final Dense layer with sigmoid activation.

### 4. GRU
The GRU model includes:
- Embedding layer with GloVe vectors.
- GRU layer with 64 units and tanh activation.
- Final Dense layer with sigmoid activation.

## Training and Evaluation
All models were trained using the Adam optimizer and binary cross-entropy loss function. We used Google Colab with a T4 GPU for training.

### Model Results:
| Model     | Units | Batch Size | Activation Function | Training Accuracy | Testing Accuracy |
|-----------|-------|------------|---------------------|-------------------|------------------|
| Shallow   | -     | 64         | tanh                | 54.71%            | 53.59%           |
| Shallow   | -     | 64         | relu                | 54.68%            | 53.53%           |
| LSTM      | 64    | 32         | tanh                | 97.02%            | 85.90%           |
| Bi-LSTM   | 64    | 32         | tanh                | 97.25%            | 86.85%           |
| GRU       | 64    | 32         | tanh                | 97.12%            | 87.14%           |

## Improvements
- The shallow model achieved lower accuracy compared to more complex models like LSTM, Bi-LSTM, and GRU.
- GRU outperformed the Bi-LSTM model with slightly better testing accuracy.
- Future improvements can be achieved by adding regularization techniques like dropout or early stopping.

## Conclusion
The GRU model performed the best with a testing accuracy of 87.14%, suggesting that it effectively captures sequential dependencies in the movie reviews. Additional improvements can be made through further hyperparameter tuning and regularization techniques.

