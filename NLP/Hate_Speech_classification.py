import csv
import io
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from nltk import TweetTokenizer, PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn import metrics
from scipy import sparse

vocab = {}
home_dir = 'D:\\UMass\\Spring20\\685\\Project\\english_dataset\\english_dataset\\'
nn_data_dir = 'D:/UMass/Spring20/685/Project/english_dataset/english_dataset/nn_data/'
embedding_dim = 0
# punctuation = r"""!"$%&'()*+,-./:;<=>?@[\]^_`{|}~"""


def load_data():
    return pd.read_csv(home_dir+'english_dataset.tsv', sep="\t")


def test_load_data():
    return pd.read_csv(home_dir+"hasoc2019_en_test-2919.tsv", sep="\t")


def preprocessor(tweet):
    global vocab
    tokenizer = TweetTokenizer()
    stops = set(stopwords.words("english"))
    # porter = PorterStemmer()
    # lemmatizer = WordNetLemmatizer()
    tweet = tweet.lower()
    tweet = re.sub(r"^RT ", ' [RT] ', tweet)
    while len(re.findall(r'@[^ :]*:?', tweet)) >= 1:
        tweet = re.sub(r'@[^ :]*:?', ' [USER] ', tweet)
    tweet = re.sub(r"http[s]?:\/\/[^ \n]*", ' [LINK] ', tweet)
    tweet = re.sub(r"\s+", ' ', tweet)
    tweet = re.sub(r'[‘’“”…]', ' <quote> ', tweet)
    tweet = re.sub(r'#', ' <hashtag> ', tweet)
    # tokens = tweet.strip().split()
    # tokens = tweet
    tokens = tokenizer.tokenize(tweet)
    puncs = str.maketrans('', '', punctuation)
    tokens = [w.translate(puncs) for w in tokens]
    tokens = [w for w in tokens if not len(w) < 1]
    tokens = [w for w in tokens if not w in stops]
    # tokens = ['[hashtag]' for w in tokens if w == ]'#']
    # tokens = [porter.stem(w) for w in tokens]
    # tokens = [lemmatizer.lemmatize(w) for w in tokens]
    for word in tokens:
        if word not in vocab.keys():
            vocab[word] = 1
        else:
            vocab[word] += 1
    return tokens


def process_data(tweet):
    # processedTweets = []
    # for tweet in tweets:
    tweet = tweet.lower()  # convert text to lower-case
    tweet = re.sub(r'\(((www\.[^\s]+)|(https?://[^\s]+))\)', ' <url>', tweet)  # replace embedded URLs with '<url>''
    tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', ' <url>', tweet)  # replace URLs with '<url>''
    tweet = re.sub(r'[ur]/[^\s]+', ' <user>', tweet)  # replace usernames and subreddits with '<user>''
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # remove the # in #hashtag
    tweet = re.sub(r'[-+]?[\.\d]*[\d]+[:,\.\d]*','<number>', tweet)  # replace numbers with '<number>''
    tweet = re.sub(r'([!?\.]){2,}', r'\1 <repeat>', tweet)
    tweet = tweet.replace(".", " . ")
    tweet = tweet.replace("?", " ? ")
    tweet = re.sub(r'#', ' <hashtag> ', tweet)
    # processedTweets.append(tweet)
    return tweet.strip().split()


def cross_val_split(X, Y):
    # Cross validation split
    stratkfold = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    inp_idx = []
    for train_ix, test_ix in stratkfold.split(X, Y):
        # train_X, test_X = X[train_ix], X[test_ix]
        # train_y, test_y = Y[train_ix], Y[test_ix]
        inp_idx.append((train_ix, test_ix))
        # train_0, train_1, train_2, train_3 = len(train_y[train_y == 0]), len(train_y[train_y == 1]), len(
        #     train_y[train_y == 2]), len(train_y[train_y == 3])
        # test_0, test_1, test_2, test_3 = len(test_y[test_y == 0]), len(test_y[test_y == 1]), len(
        #     test_y[test_y == 2]), len(test_y[test_y == 3])
        # print('>Train: 0=%d, 1=%d, 2=%d, 3=%d Test: 0=%d, 1=%d, 2=%d, 3=%d' %
        #       (train_0, train_1, train_2, train_3, test_0, test_1, test_2, test_3))
    return inp_idx


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def load_embedding(embedding_type):
    global embedding_dim
    embeddings_index = {}
    f = None
    if embedding_type == 'glove':
        f = open('D:/workspace/glove.twitter.27B/glove.twitter.27B.200d.txt', encoding="utf8")
        embedding_dim = 200
    elif embedding_type == 'fasttext':
        f = open('D:/workspace/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec', encoding="utf8")
        embedding_dim = 300
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float64')
            embeddings_index[word] = coefs
        except ValueError:
            pass
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def vec_emb(toks, embed_matrix):
    M = []
    for w in toks:
        try:
            M.append(embed_matrix[w])
        except:
            continue
    M = np.asarray(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(embedding_dim)
    return v / np.sqrt((v ** 2).sum())


def get_labels(lab):
    Y = []
    for label in lab:
        if label == 'NONE':
            Y.append(0)
        elif label == 'HATE':
            Y.append(1)
        elif label == 'PRFN':
            Y.append(2)
        elif label == 'OFFN':
            Y.append(3)
    Y = pd.Series(Y)
    return Y


def process_inputs(word_model, write_flag):
    data = load_data()
    test_data = test_load_data()
    x = data.copy(deep=True)
    x_test = test_data.copy(deep=True)
    # x.text = x.text.apply(preprocessor)
    # x_test.text = x_test.text.apply(preprocessor)
    x.text = x.text.apply(process_data)
    x_test.text = x_test.text.apply(process_data)
    print(x.groupby(['task_2']).count())
    X = x['text']
    X_test = x_test['text']
    Y = get_labels(x['task_2'])
    Y_test = get_labels(x_test['task_2'])
    if write_flag:
        frame = {'label': Y, 'text': x.text.apply(' '.join)}
        frame_test = {'label': Y_test, 'text': x_test.text.apply(' '.join)}
        df = pd.DataFrame(frame)
        df_test = pd.DataFrame(frame_test)
        print(df)
        print(df_test)
        df.to_csv(nn_data_dir+'bert_hasoc_train.txt', sep=',', index=False, header=False)
        df_test.to_csv(nn_data_dir + 'bert_hasoc_test.txt', sep=',', index=False, header=False)
    if word_model == 'WE':
        embed_matrix = load_embedding('fasttext')
        X = np.vstack([vec_emb(x1, embed_matrix) for x1 in tqdm(X)])
        X = preprocessing.scale(X)
        X_test = np.vstack([vec_emb(x1, embed_matrix) for x1 in tqdm(X_test)])
        X_test = preprocessing.scale(X_test)
    elif word_model == 'BOW':
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(x.text.apply(' '.join))
        X_test = vectorizer.transform(x_test.text.apply(' '.join))
    elif word_model == 'TFIDF':
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(x.text.apply(' '.join))
        X_test = vectorizer.transform(x_test.text.apply(' '.join))
    print(X.shape, Y.shape, X_test.shape, Y_test.shape)
    return X, Y, X_test, Y_test


def log_reg(X, Y, s_idx, X_t, Y_t):
    avg_f = 0.0
    best_f = -1
    best_model = None
    conf_mat = None
    class_rep = None
    for (train_idx, test_idx) in s_idx:
        logreg = LogisticRegression(penalty='l2', tol=0.0001, C=1.0, random_state=0, solver='lbfgs',
                                    multi_class='auto', max_iter=1000)
        logreg.fit(X[train_idx], Y[train_idx])
        pred_y = logreg.predict(X[test_idx])
        # acc = logreg.score(X[test_idx], Y[test_idx])
        fs = f1_score(Y[test_idx], pred_y, average='macro')
        avg_f += fs
        if fs > best_f:
            best_f = fs
            best_model = logreg
            conf_mat = metrics.confusion_matrix(Y[test_idx], pred_y)
            class_rep = metrics.classification_report(Y[test_idx], pred_y, digits=3)
    avg_f /= 10
    print("Training metrics...")
    print("Average Val f1 score : ", avg_f)
    print("Best Val f1 score : ", best_f)
    print(conf_mat)
    print(class_rep)
    # Running on test
    test_y_predict = best_model.predict(X_t)
    # test_acc = best_model.score(X_t, Y_t)
    pred_y_test = best_model.predict(X_t)
    test_f = f1_score(Y_t, pred_y_test, average='macro')
    test_conf_mat = metrics.confusion_matrix(Y_t, test_y_predict)
    test_class_rep = metrics.classification_report(Y_t, test_y_predict, digits=3)
    print("Test metrics....")
    print("Final Test f1 score : ", test_f)
    print(test_conf_mat)
    print(test_class_rep)
    return best_model


def clean_data():
    fil_train = 'D:/UMass/Spring20/685/Project/Reddit/reddit_train.txt'
    fil_test = 'D:/UMass/Spring20/685/Project/Reddit/reddit_test.txt'
    with open(fil_test, 'r', encoding="UTF-8") as datafile:
        data = [line.strip().split(',', maxsplit=1) for line in datafile]
        data_text = np.array(list(map(lambda x: x[1], data)))
        data_label = np.array(list(map(lambda x: x[0].strip(), data)))
        idx = []
        for i in range(len(data_text)):
            if len(data_text[i]) <= 150:
                idx.append(i)
        print(len(idx))
        print(len(data_text[idx]))
        print(len(data_label[idx]))
    full_df = pd.DataFrame({"label": data_label[idx], "text": data_text[idx]})
    full_df.to_csv('D:/UMass/Spring20/685/Project/Reddit/mod_reddit_test.txt', sep=",",
                   header=False, index=False, encoding="UTF-8", quoting=None)
    return True


def main():
    X_train, Y_train, X_test, Y_test = process_inputs('BOW', False)
    # sparse.save_npz(nn_data_dir + "hasoc_tr.npz", X_train)
    # sparse.save_npz(nn_data_dir + "hasoc_te.npz", X_test)
    split_idx = cross_val_split(X_train, Y_train)
    model = log_reg(X_train, Y_train, split_idx, X_test, Y_test)
    # weights = model.coef_
    # print(weights.shape)
    # wt0 = weights[0]
    # wt1 = weights[0]
    # wt2 = weights[0]
    # wt3 = weights[0]
    #
    # class_0 = []
    # class_1 = []
    # class_2 = []
    # class_3 = []

    # for wrd in vocab.keys():
    #     try:
    #         wrd_emb = vocab[wrd]
    #     except:
    #         wrd_emb = np.ones(200)

    # clean_data()
    exit(0)
    return


if __name__ == "__main__":
    main()