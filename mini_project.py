from email.policy import default
import warnings
warnings.filterwarnings('ignore')

# Modules for data manipulation
import numpy as np
import pandas as pd
import re

# Modules for visualization
import matplotlib.pyplot as plt
import seaborn as sb

# Tools for preprocessing input data
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tools for creating ngrams and vectorizing input data
from gensim.models import Word2Vec, Phrases

# Tools for building a model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

# Tools for assessing the quality of model prediction
from sklearn.metrics import accuracy_score, confusion_matrix

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIG_SIZE = 16
LARGE_SIZE = 20

params = {
    'figure.figsize': (16, 8),
    'font.size': SMALL_SIZE,
    'xtick.labelsize': MEDIUM_SIZE,
    'ytick.labelsize': MEDIUM_SIZE,
    'legend.fontsize': BIG_SIZE,
    'figure.titlesize': LARGE_SIZE,
    'axes.titlesize': MEDIUM_SIZE,
    'axes.labelsize': BIG_SIZE
}
plt.rcParams.update(params)

usecols = ['sentiment','text_dat']
# col_names_twit = [""]
train_data = pd.read_csv(
    filepath_or_buffer='../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip',
    usecols=usecols, sep='\t')
train_data.columns = ["sentiment","text_dat"]
unlabeled_data = pd.read_csv(
    filepath_or_buffer="../input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip", 
    error_bad_lines=False,
    sep='\t')
unlabeled_data.columns = ["sentiment","text_dat"]

twit_test = pd.read_csv("../input/sentiment-140/test_twit.csv",header=None)
twit_test.columns = ['sentiment','id','date','query','user','text_dat']
twit_test = twit_test.drop(columns=['id', 'date','query','user'])
twit_train = pd.read_csv("../input/sentiment-140/train_twit.csv", encoding='latin-1',header=None)
twit_train.columns = ['sentiment','id','date','query','user','text_dat']
twit_train = twit_train.drop(columns=['id', 'date','query','user'])


twit_train['sentiment'] = np.where(twit_train['sentiment'] == 2, 0.5, twit_train['sentiment'])
twit_train['sentiment'] = np.where(twit_train['sentiment'] == 4, 1, twit_train['sentiment'])

twit_test['sentiment'] = np.where(twit_test['sentiment'] == 2, 0.5, twit_test['sentiment'])
twit_test['sentiment'] = np.where(twit_test['sentiment'] == 4, 1, twit_test['sentiment'])

datasets = [train_data, unlabeled_data, twit_test, twit_train]
titles = ['Train data', 'Submission data', 'Unlabeled train data','twit_test', 'twit_train']
for dataset, title in zip(datasets,titles):
    print(title)
    dataset.info()


all_text_dats = np.array([], dtype=str)
for dataset in datasets:
    all_text_dats = np.concatenate((all_text_dats, dataset.text_dat), axis=0)
print('Total number of text_dats:', len(all_text_dats))

plt.hist(train_data[train_data.sentiment == 1].sentiment,
         bins=2, color='green', label='Positive')
plt.hist(train_data[train_data.sentiment == 0].sentiment,
         bins=2, color='blue', label='Negative')
plt.hist(train_data[train_data.sentiment == 0.5].sentiment,
         bins=2, color='yellow', label='Neutral')
plt.title('Classes distribution in the train data', fontsize=LARGE_SIZE)
plt.xticks([])
plt.xlim(-0.5, 2)
plt.legend()
plt.savefig("/graphs_and_results/histogram_sentiment.png")

def clean_text_dat(raw_text_dat: str) -> str:
    # 1. Remove HTML
    text_dat_text = BeautifulSoup(raw_text_dat, "lxml").get_text()
    # 2. Remove non-letters
    text_dat_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text_dat_text, flags=re.MULTILINE)
    text_dat_text = re.sub(r'^@?(\w){1,15}$', '', text_dat_text, flags=re.MULTILINE)
    letters_only = REPLACE_WITH_SPACE.sub(" ", text_dat_text)
    # 3. Convert to lower case
    lowercase_letters = letters_only.lower()
    return lowercase_letters


def lemmatize(tokens: list) -> list:
    # 1. Lemmatize
    tokens = list(map(lemmatizer.lemmatize, tokens))
    lemmatized_tokens = list(map(lambda x: lemmatizer.lemmatize(x, "v"), tokens))
    # 2. Remove stop words
    meaningful_words = list(filter(lambda x: not x in stop_words, lemmatized_tokens))
    return meaningful_words


def preprocess(text_dat: str, total: int, show_progress: bool = True) -> list:
    if show_progress:
        global counter
        counter += 1
        print('Processing... %6i/%6i'% (counter, total), end='\r')
    # 1. Clean text
    text_dat = clean_text_dat(text_dat)
    # 2. Split into individual words
    tokens = word_tokenize(text_dat)
    # 3. Lemmatize
    lemmas = lemmatize(tokens)
    # 4. Join the words back into one string separated by space,
    # and return the result.
    return lemmas

counter = 0
REPLACE_WITH_SPACE = re.compile(r'[^A-Za-z\s]')
stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()

all_text_dats = np.array(list(map(lambda x: preprocess(x, len(all_text_dats)), all_text_dats)))
counter = 0

X_train_data = all_text_dats[:train_data.shape[0]]
Y_train_data = train_data.sentiment.values

train_data['text_dat_length'] = np.array(list(map(len, X_train_data)))
median = train_data['text_dat_length'].median()
mean = train_data['text_dat_length'].mean()
mode = train_data['text_dat_length'].mode()[0]

fig, ax = plt.subplots()
ax.set_facecolor('floralwhite')
sb.distplot(train_data['text_dat_length'], bins=train_data['text_dat_length'].max(),
            hist_kws={"alpha": 0.9, "color": "goldenrod"}, ax=ax,
            kde_kws={"color": "navy", 'linewidth': 3})
ax.set_xlim(left=0, right=np.percentile(train_data['text_dat_length'], 95))
ax.set_xlabel('Words in text_dat')
ymax = 0.014
plt.ylim(0, ymax)
ax.plot([mode, mode], [0, ymax], '--', label=f'mode = {mode:.2f}', linewidth=4, color = 'maroon')
ax.plot([mean, mean], [0, ymax], '--', label=f'mean = {mean:.2f}', linewidth=4)
ax.plot([median, median], [0, ymax], '--',
        label=f'median = {median:.2f}', linewidth=4)
ax.set_title('Words per text_dat distribution', fontsize=20)
plt.legend()
plt.savefig("/graphs_and_results/words_distribution.png")

bigrams = Phrases(sentences=all_reviews)

trigrams = Phrases(sentences=bigrams[all_reviews])

embedding_vector_size = 256
trigrams_model = Word2Vec(
    sentences = trigrams[bigrams[all_reviews]],
    size = embedding_vector_size,
    min_count=3, window=5, workers=4)
trigrams_model.summary()

print("Vocabulary size:", len(trigrams_model.wv.vocab))

def vectorize_data(data, vocab: dict) -> list:
    print('Vectorize sentences...', end='\r')
    keys = list(vocab.keys())
    filter_unknown = lambda word: vocab.get(word, None) is not None
    encode = lambda review: list(map(keys.index, filter(filter_unknown, review)))
    vectorized = list(map(encode, data))
    print('Vectorize sentences... (done)')
    return vectorized

print('Convert sentences to sentences with ngrams...', end='\r')
X_data = trigrams[bigrams[X_train_data]]
print('Convert sentences to sentences with ngrams... (done)')
input_length = 150
X_pad = pad_sequences(
    sequences=vectorize_data(X_data, vocab=trigrams_model.wv.vocab),
    maxlen=input_length,
    padding='post')
print('Transform sentences to sequences... (done)')

X_train, X_test, y_train, y_test = train_test_split(
    X_pad,
    Y_train_data,
    test_size=0.05,
    shuffle=True,
    random_state=42)

def build_model(embedding_matrix: np.ndarray, input_length: int):
    model = Sequential()
    model.add(Embedding(
        input_dim = embedding_matrix.shape[0],
        output_dim = embedding_matrix.shape[1], 
        input_length = input_length,
        weights = [embedding_matrix],
        trainable=False))
    model.add(Bidirectional(LSTM(128, recurrent_dropout=0.1)))
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model

model = build_model(
    embedding_matrix=trigrams_model.wv.vectors,
    input_length=input_length)

model.compile(
    loss="binary_crossentropy",
    optimizer='adam',
    metrics=['accuracy'])

history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    batch_size=100,
    epochs=100)

def plot_confusion_matrix(y_true, y_pred, ax, class_names, vmax=None,
                          normed=True, title='Confusion matrix'):
    matrix = confusion_matrix(y_true,y_pred)
    if normed:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    sb.heatmap(matrix, vmax=vmax, annot=True, square=True, ax=ax,
               cmap=plt.cm.Blues_r, cbar=False, linecolor='black',
               linewidths=1, xticklabels=class_names)
    ax.set_title(title, y=1.20, fontsize=16)
    ax.set_ylabel('True labels', fontsize=12)
    ax.set_xlabel('Predicted labels', y=1.10, fontsize=12)
    ax.set_yticklabels(class_names, rotation=0)
    plt.savefig("/graphs_and_results/confusion.png")


y_train_pred = model.predict(X_train)
y_train_pred = np.array(y_train_pred)
choices = [1,0.5,0]
conditions = [
    (y_train_pred > 0.6),
    (y_train_pred <= 0.6) & (y_train_pred > 0.3),
    (y_train_pred <= 0.3)
]
y_train_pred = np.select(conditions,choices)

y_test_pred = model.predict(X_test)
conditions = [
    (y_test_pred > 0.6),
    (y_test_pred <= 0.6) & (y_test_pred > 0.3),
    (y_test_pred <= 0.3)
]
y_test_pred = np.select(conditions,choices)


fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2)
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=1)

# fig.tight_layout()

plot_confusion_matrix(y_train, y_train_pred, ax=axis1,
                      title='Confusion matrix (train data)',
                      class_names=['Positive', 'Negative'])

plot_confusion_matrix(y_test, y_test_pred, ax=axis2,
                      title='Confusion matrix (test data)',
                      class_names=['Positive', 'Negative'])





fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

# summarize history for accuracy
axis1.plot(history.history['acc'], label='Train', linewidth=3)
axis1.plot(history.history['val_acc'], label='Validation', linewidth=3)
axis1.set_title('Model accuracy', fontsize=16)
axis1.set_ylabel('accuracy')
axis1.set_xlabel('epoch')
axis1.legend(loc='upper left')
plt.savefig("/graphs_and_results/accuracy_epoch.png")

# summarize history for loss
axis2.plot(history.history['loss'], label='Train', linewidth=3)
axis2.plot(history.history['val_loss'], label='Validation', linewidth=3)
axis2.set_title('Model loss', fontsize=16)
axis2.set_ylabel('loss')
axis2.set_xlabel('epoch')
axis2.legend(loc='upper right')
plt.savefig("/graphs_and_results/loss_epoch.png")

