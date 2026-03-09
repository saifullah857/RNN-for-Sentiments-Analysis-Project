# IMDB Sentiment Analysis using RNN (PyTorch)

## Project Overview

This project performs **Sentiment Analysis on the IMDB Movie Reviews dataset** using **Natural Language Processing (NLP)** techniques and a **Recurrent Neural Network (RNN)** implemented in **PyTorch**.

The goal of the model is to classify movie reviews as **positive** or **negative** based on their textual content.

---

# Dataset

The project uses the **IMDB Dataset of 50,000 movie reviews** containing:

* Review text
* Sentiment label (positive / negative)

After preprocessing, sentiment labels are converted into numeric values:

* `1 → Positive`
* `0 → Negative`

---

# Project Pipeline

The workflow of the project includes the following steps:

## 1. Data Loading

The dataset is loaded using **Pandas**.

```
df = pd.read_csv("IMDB Dataset.csv")
```

---

## 2. Data Cleaning

Several preprocessing techniques are applied to clean the text data:

* Convert text to lowercase
* Remove URLs
* Remove punctuation
* Remove HTML tags
* Remove duplicate records

---

## 3. Text Preprocessing (NLP)

### Tokenization

Text is split into individual words using **NLTK**.

### Stopword Removal

Common words such as *the, is, in, and* are removed because they do not add meaning.

### Stemming

Words are reduced to their root form using **Porter Stemmer**.

Example:

```
running → run
playing → play
```

---

## 4. Text Vectorization

The cleaned text is converted into numerical form using **TF-IDF Vectorizer**.

```
TfidfVectorizer(max_features=5000)
```

This converts text into a **feature matrix** suitable for machine learning models.

---

## 5. Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

```
train_test_split(X, y, test_size=0.2)
```

---

## 6. PyTorch Dataset and DataLoader

The TF-IDF features are converted into **PyTorch tensors** and loaded using:

* `TensorDataset`
* `DataLoader`

This allows efficient batch training.

---

# Model Architecture

The model uses a **Recurrent Neural Network (RNN)**.

### Architecture

Input Layer
↓
RNN Layer
↓
Fully Connected Layer
↓
Sigmoid Activation
↓
Binary Output (0 or 1)

### Model Structure

```
RNN(
    input_size = number_of_features,
    hidden_size = 128,
    num_layers = 1
)
```

---

# Training

Training uses:

* **Loss Function:** Binary Cross Entropy Loss (`BCELoss`)
* **Optimizer:** Adam
* **Epochs:** 10
* **Batch Size:** 64

Training loop performs:

1. Forward pass
2. Loss calculation
3. Backpropagation
4. Weight update

---

# Model Evaluation

After training, the model is evaluated on the **test dataset**.

Accuracy is calculated using:

```
accuracy = correct_predictions / total_predictions
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-Learn
* PyTorch
* Jupyter Notebook

---

# Project Structure

```
project-folder/
│
├── IMDB Dataset.csv
├── RNN-code.ipynb
├── README.md
```

---

# How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/your-username/repository-name.git
```

### 2. Install required libraries

```
pip install pandas numpy nltk scikit-learn torch
```

### 3. Run the notebook

Open the Jupyter notebook and execute all cells.

---

# Results

The model learns to classify movie reviews and predicts whether a review is **positive or negative** based on text content.

---

# Future Improvements

Possible improvements for the project:

* Use **LSTM or GRU instead of simple RNN**
* Implement **Word Embeddings (Word2Vec / GloVe)**
* Use **Deep Learning Transformers (BERT)**
* Improve preprocessing pipeline

---

# Author

Saif Ullah
