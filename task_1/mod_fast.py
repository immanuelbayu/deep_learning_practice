import numpy as np
import pandas as pd
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models import FastText
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

sentimen_data = pd.read_csv('../data/Indonesian Sentiment Twitter Dataset Labeled.csv', sep="\t")
sentimen_data.head()

alay_dict = pd.read_csv("../data/new_kamusalay.csv", encoding = 'latin-1', header=None)
alay_dict = alay_dict.rename(columns=  {0: 'original',
                                        1: 'replacement'})
stopword_dict = pd.read_csv('../data/stopwordbahasa.csv', header=None)
stopword_dict = stopword_dict.rename(columns={0: 'stopword'})

sentimen_data.isnull().sum()

factory = StemmerFactory()
stemmer = factory.create_stemmer()


# lowercase
def lowercase(text):
    return text.lower()


def remove_unnecessary_char(text):
    text = re.sub('\n', ' ', text)  # Remove every '\n'
    text = re.sub('rt', ' ', text)  # Remove every retweet symbol
    text = re.sub('((@[^\s]+)|(#[^\s]+))', ' ', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', text)  # Remove every URL
    text = re.sub('  +', ' ', text)  # Remove extra spaces
    return text


def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    return text


alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))


def normalize_alay(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])


def remove_stopword(text):
    text = ' '.join(['' if word in stopword_dict.stopword.values else word for word in text.split(' ')])
    text = re.sub('  +', ' ', text)  # Remove extra spaces
    text = text.strip()
    return text


def stemming(text):
    return stemmer.stem(text)


def preprocess(text):
    text = lowercase(text)  # 1
    text = remove_unnecessary_char(text)  # 2
    text = remove_nonaplhanumeric(text)  # 3
    text = normalize_alay(text)  # 4
    text = remove_stopword(text)  # 5
    text = stemming(text)  # 6
    return text

sentimen_data.replace(-1, 0, inplace=True)

tweet = sentimen_data['Tweet'].apply(preprocess)

sentimen_data.head()

tweets = sentimen_data['Tweet'].values
sentiments = sentimen_data['sentimen'].values

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)
sequences = tokenizer.texts_to_sequences(tweets)

# Pad sequences
max_seq_length = max(len(x) for x in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length)

# Train FastText
model_ft = FastText([seq.split() for seq in tweets], vector_size=150, window=5, min_count=3, epochs=100)

# Create embedding matrix
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((vocab_size, 150))
for word, i in tokenizer.word_index.items():
    try:
        embedding_vector = model_ft.wv[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        continue

# Build LSTM model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=150, weights=[embedding_matrix], input_length=max_seq_length, trainable=False))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Convert sentiments to binary labels
binary_labels = np.array([1 if x == 1 else 0 for x in sentiments])

# Train model
model.fit(padded_sequences, binary_labels, epochs=1, batch_size=1)