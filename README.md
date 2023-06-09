# **Military Prohibited Messages Detector for the War in Ukraine**
## **Project Description**

The "Military Prohibited Messages Detector for the War in Ukraine" is an innovative project aimed at enhancing the security and efficiency of military operations in the ongoing conflict in Ukraine. In the era of advanced communication technologies, it has become crucial to monitor and intercept unauthorized or prohibited messages to maintain operational secrecy, protect sensitive information, and ensure the safety of military personnel.

The primary objective of this project is to develop a system capable of  detecting and analyzing prohibited messages exchanged within communication channels, like social media. 

## **Datasets Description**
The "Military Prohibited Messages Detector for the War in Ukraine" project relies on two essential datasets: one containing legitimate, non-prohibited messages (referred to as "ham" messages) and another comprising forbidden messages. These datasets serve as the foundation for training and fine-tuning the detection models.

## **Preprocessing stage**
During the preprocessing stage, the datasets undergo several crucial steps to ensure optimal performance and accurate classification. First, the text data is subjected to tokenization, where sentences are divided into individual words or tokens. This process enables the models to understand the messages at a more granular level and capture the nuances of language.

Following tokenization, the texts undergo various text cleaning operations. These operations involve removing unnecessary elements such as punctuation marks, special characters, and numbers, as they do not contribute significantly to the detection of prohibited messages. Additionally, any extraneous white spaces or irrelevant formatting is eliminated to streamline the data.

Next, the text is subjected to lowercasing, converting all characters to lowercase. This step is important to ensure consistency in the data, as it eliminates the potential discrepancy between uppercase and lowercase letters when the models learn patterns and linguistic cues.

After preprocessing, the datasets are divided into training and validation sets. The training set is used to train the models on a large volume of labeled examples, while the validation set is employed to fine-tune and evaluate the model's performance during training.

## **Chosen Models**
In the "Military Prohibited Messages Detector for the War in Ukraine" project, three different deep learning models are employed: Decision Tree Classifier (DTC), Recurrent Neural Network (RNN), and Long Short-Term Memory (LSTM) networks.

## **DTC Description**
The DTC is a supervised learning algorithm that uses a decision tree structure to classify messages based on a set of predetermined rules. It serves as an initial model for binary classification, categorizing messages as either "ham" or "forbidden" based on specific criteria.

## **RNN and LSTM Description**
The RNN and LSTM models are particularly well-suited for analyzing sequential data, making them ideal for detecting prohibited messages within text. These models can capture the contextual dependencies and patterns in the messages, enabling them to identify the subtle indicators of forbidden content. The RNN and LSTM models are trained and fine-tuned using the preprocessed datasets, optimizing their ability to classify messages accurately.


## **Model Training and Evaluation**


1. Decision Tree Classifier (DTC):
  - Train the DTC model on the training set.
  - Perform hyperparameter optimization using  grid search technique to find the best combination of hyperparameters.
  - Evaluate the DTC model on the validation set to assess its performance.

2. Recurrent Neural Network (RNN):
  - Preprocess the dataset for LSTM input, including any necessary embedding or encoding techniques.
  - Train the RNN model on the training set, adjusting hyperparameters such as the number of layers, hidden units, and learning rate using BayesSearch.
  - Evaluate the RNN model on the validation set to measure its performance.

3. Long Short-Term Memory (LSTM):
  - Preprocess the dataset for LSTM input, including any necessary embedding or encoding techniques.
  - Train the LSTM model on the training set, tuning hyperparameters like the number of layers, hidden units, dropout rates, and learning rate  using BayesSearch.
  - Evaluate the LSTM model's performance on the validation set.

## **Installation**

  - Clone the repository:

`git clone ###`

  - Install the required dependencies:

`pip install -r requirements.txt`

## **Usage**

### **IPYNB**

In the section **"Змінні для роботи"** set a boolean value if you work in Colab

If you are working in Colab:
-  specify the path to saved datasets on your Google Drive.
-  specify the path for saving preprocessed datasets on your Google Drive.
-  specify the path for saving models on your Google Drive.

Run the nn_cp.ipynb: Run All

### **Telegram bot**

Run the telebot.py

Access the telegram_bot from Telegram:

https://t.me/peacekeeperLpnu_bot


