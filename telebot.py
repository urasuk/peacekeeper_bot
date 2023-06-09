import telebot
import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
import re
import string

bot = telebot.TeleBot('6016023163:AAGMSCQtW0YsspvNwSH1nJGBYEig_9q2sqM')
df_prepros_train = pd.read_csv('saved_datasets/df_preprossed_train.csv')
df_prepros_test = pd.read_csv('saved_datasets/df_preprossed_test.csv')
X_train = list(df_prepros_train['message'])
y_train = list(df_prepros_train['label'])
X_test = list(df_prepros_test['message'])
y_test = list(df_prepros_test['label'])


model = tf.keras.models.load_model('saved_models/LSTM_best.tf')


DTC_param = [100, 2]
RNN_param = [None, 2]
LSTM_param = [400, 2]
vectorizer = tf.keras.layers.TextVectorization(max_tokens=LSTM_param[0], ngrams=LSTM_param[1])
vectorizer.adapt(np.array(X_train))
train_tokenized = vectorizer(X_train).numpy()
test_tokenized = vectorizer(X_test).numpy()

tokenizerRegularExpression = nltk.RegexpTokenizer(r"\w+")

def preprocess_twitter_dataset(tokenizer, datasetInput):
    dataset = datasetInput.copy()
    dataset['message'] = dataset['message'].str.lower()
    dataset['message'] = dataset['message'].apply(lambda x: re.sub(r"http\S+|www\S+|https\S+", "", x))
    dataset['message'] = dataset['message'].apply(lambda x: re.sub(r"@\w+|#\w+", "", x))
    dataset['message'] = dataset['message'].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
    dataset['message'] = dataset['message'].apply(tokenizer.tokenize)
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    dataset['message'] = dataset['message'].apply(lambda x: [word for word in x if word not in stop_words])
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    dataset['message'] = dataset['message'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    return dataset


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Hello! I'm a bot for identifying military-prohibited messages.")
    bot.send_message(message.chat.id, "Below are examples of military prohibited messages:"
                     +"\n\n - In the city of Bakhmut, the military deployed a base for storing heavy military equipment, which includes tanks, armored personnel carriers and artillery."+
                     "\n\n - Just at 6:30 p.m., there were reports of rocket fire on Chernivtsi, probably from an 'Iskander' missile." + "\n\n - The Bandera group has set up a new checkpoint close to the town of Torez. The coordinates are 48.012137, 38.391937."+"\n\n - A batch of 10 Osa-AKM anti-aircraft missile systems has arrived at the military base in the city of Dnipro, which strengthens the air defense capabilities of Ukraine.")
    bot.send_message(message.chat.id, "Please write a message for verification.")

@bot.message_handler(func=lambda message: True)
def process_message(message):
    text = message.text
    df = preprocess_twitter_dataset(tokenizerRegularExpression,pd.DataFrame({'message':[text]}))
    text = df.iloc[0, 0]
    text = ' '.join(text)
    prediction = model.predict([text])
    predicted_class = np.argmax(prediction)
    if predicted_class == 1:
        bot.send_message(message.chat.id, "This is a military-prohibited message.")
    else:
        bot.send_message(message.chat.id, "This is not a military-prohibited message.")
bot.polling(none_stop=True)
