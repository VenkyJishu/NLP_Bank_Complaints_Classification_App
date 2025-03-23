# Customer Complaint Classification App

This project aims to classify customer complaints into predefined categories using Natural Language Processing (NLP) and machine learning. The application preprocesses complaint data, performs feature extraction, and uses machine learning models to classify complaints into the following categories:

- **Bank Account services**
- **Credit card or prepaid card**
- **Others**
- **Theft/Dispute Reporting**
- **Mortgage/Loan**

The app is deployed in Huggingface using **Gradio** to provide an easy-to-use interface for classifying complaints.

## Features

- **Text Preprocessing**:
  - Tokenization, Lemmatization, and Part-of-Speech (POS) tagging using **NLTK**.
  - **Seaborn** for visualizing complaint character lengths.
  - N-gram (Unigram, Bigram, Trigram) creation to capture word sequence patterns.
  - **Word Cloud** visualization to highlight the most frequent words in complaints.

- **Feature Extraction**:
  - **TF-IDF Vectorization** to convert the cleaned complaint text into numerical features.
  - **CountVectorizer** and **TF-IDF Transformer** to prepare the data for model training.

- **Topic Modeling**:
  - **Non-negative Matrix Factorization (NMF)** is used for topic modeling to discover hidden topics in the complaint data.

- **Model Evaluation and Training**:
  - Evaluating several models: **Naive Bayes**, **Logistic Regression**, **Random Forest**, and **Decision Tree**.
  - **Logistic Regression** was selected based on its superior performance metrics (accuracy, precision, recall, F1-score).

- **Prediction and Deployment**:
  - **Gradio** interface for deploying the app, allowing users to input complaints and get predicted categories.

## Installation

Follow the instructions below to run this project locally:

### 1. Clone the repository:

git clone https://github.com/yourusername/bank_complaints_classification_app.git
cd bank_complaints_classification_app

### 2. Set up a virtual environment:
Create a virtual environment using venv:

python -m venv jishu_env
Activate the virtual environment:

Windows:

.\jishu_env\Scripts\activate

Mac/Linux:

source jishu_env/bin/activate
### 3. Install dependencies:
Install the required libraries using:


pip install -r requirements.txt

### Preprocessing
The preprocessing steps applied to the customer complaint text include:

### Tokenization: Breaking the text into individual words or tokens.
### Lemmatization: Converting words to their base or root form using NLTK.
### POS Tagging: Assigning Part-of-Speech labels to each token using NLTK.
### Complaints Character Length Visualization: Visualizing the distribution of complaint lengths using Seaborn.

### N-grams:

Unigrams, Bigrams, and Trigrams are created to capture the frequency of word sequences in the complaint texts.
Word Cloud:

Visualizing the most frequent words in the complaint texts using a Word Cloud to gain insights into common complaint themes.

### Feature Extraction
**TF-IDF Vectorization** is applied to convert the complaint text into numerical features that can be used by machine learning models.

The models are trained using the CountVectorizer and TF-IDF Transformer classes from scikit-learn.

### Topic Modeling with NMF
**Non-negative Matrix Factorization (NMF)** is used for topic modeling. NMF helps to extract hidden topics from the complaint texts and create a topic column.

The topic column is combined with the existing cleaned complaint column to form the final dataset, training_data_df.

### Model Training
The training_data_df dataframe is split into training and test sets. We evaluate the following machine learning models:

### Naive Bayes
### Logistic Regression
### Random Forest
### Decision Tree
    
    After evaluating the models, Logistic Regression is selected due to its superior performance based on evaluation metrics such as accuracy, precision, recall, and F1-score.

### Prediction and Deployment
Once the model is trained, we use Gradio to deploy the app. The Gradio interface allows users to input a complaint and get predictions on the complaint category.

The models and vectorizers (CountVectorizer and TF-IDF) are saved using pickle for future use. These models are loaded when the app is used for prediction.

### Running the Application
To run the app locally:

Ensure that your virtual environment is activated.

## Run the following command:

python app.py

This will start a local Gradio interface. You can input complaint text, and the app will predict which category the complaint belongs to.

### Deployment
    The app can be deployed to the cloud using Hugging Face Spaces. You can push your code to GitHub and link it to a new Hugging Face Space for automatic deployment.

For more details on deploying the app on Hugging Face, check out the Hugging Face Spaces documentation.

### Dependencies
Python 3.x
Gradio: For creating the interactive user interface.
scikit-learn: For machine learning models and vectorization.
NLTK: For text preprocessing (tokenization, lemmatization, POS tagging).
Seaborn: For visualizing the distribution of complaint character lengths.
WordCloud: For generating word clouds.
Pickle: For saving and loading models.
Dependencies are listed in the requirements.txt file.
