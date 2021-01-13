import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
import sqlite3 as sql
from sklearn.metrics import classification_report#, accuracy_score, precision_score, recall_score,
import pickle, time

# database to train from
src = '../GiantDB/giant.db'

# connect to sqlite database
conn, cur = None, None
try:
    conn = sql.connect(src)
except sql.Error as e:
    print(e)
    sys.exit(1)

# load all the data into pandas dataframe
df = pd.read_sql_query("select distinct(title), yes_no from giant_jobs", conn)
#keys = ['title', 'yes_no'] # -- These are the ones that I am looking at

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(df['title'], df['yes_no'], test_size=0.3, stratify=df['yes_no'])
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')

# transform the data into a bunch of numbers
X_train_cv = cv.fit_transform(X_train)
X_test_cv  = cv.transform(X_test)
word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())

# create the model and fit and train
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)

# start predicting using the model
start = time.time()
predictions = naive_bayes.predict(X_test_cv)

print("Time it took:", time.time()-start) # under a second for the testing set
# print the classification report (accuracy, precision, recall, f1)
# information on what each column means:
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
# the 0 class represents: not a job title
# the 1 class represents: a job title
print(classification_report(y_test, predictions))

## example for testing on separate data
# strings = ["time", "Finance"]
# strings_cv = cv.transform(strings)
# print(type(X_test_cv), type(strings_cv))
# print(naive_bayes.predict(strings_cv))

## save the model's weights to use with other files and prevent having to train again
pkl_mod = "naive_bayes.pickle"
with open(pkl_mod, 'wb') as file:
    pickle.dump(naive_bayes, file)

pkl_cv = "nb_counter.pickle"
with open(pkl_cv, 'wb') as file:
    pickle.dump(cv, file)

# to test the data, uncomment below

# nb = "naive_bayes.pickle"
# with open(nb, 'rb') as file:
#     pickle_model = pickle.load(file)

## Testing the pickled model. IT WORKS!
# preds = pickle_model.predict(X_test_cv)
# print(preds)
# print("Just printed out the second set of predictions.")
# print(False in (preds == predictions))
=======
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
import sqlite3 as sql
from sklearn.metrics import classification_report#, accuracy_score, precision_score, recall_score,
import pickle, time

# database to train from
src = '../GiantDB/giant.db'

# connect to sqlite database
conn, cur = None, None
try:
    conn = sql.connect(src)
except sql.Error as e:
    print(e)
    sys.exit(1)

# load all the data into pandas dataframe
df = pd.read_sql_query("select distinct(title), yes_no from giant_jobs", conn)
#keys = ['title', 'yes_no'] # -- These are the ones that I am looking at

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(df['title'], df['yes_no'], test_size=0.3, stratify=df['yes_no'])
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')

# transform the data into a bunch of numbers
X_train_cv = cv.fit_transform(X_train)
X_test_cv  = cv.transform(X_test)
word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())

# create the model and fit and train
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)

# start predicting using the model
start = time.time()
predictions = naive_bayes.predict(X_test_cv)

print("Time it took:", time.time()-start) # under a second for the testing set
# print the classification report (accuracy, precision, recall, f1)
# information on what each column means:
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
# the 0 class represents: not a job title
# the 1 class represents: a job title
print(classification_report(y_test, predictions))

## example for testing on separate data
# strings = ["time", "Finance"]
# strings_cv = cv.transform(strings)
# print(type(X_test_cv), type(strings_cv))
# print(naive_bayes.predict(strings_cv))

## save the model's weights to use with other files and prevent having to train again
pkl_mod = "naive_bayes.pickle"
with open(pkl_mod, 'wb') as file:
    pickle.dump(naive_bayes, file)

pkl_cv = "nb_counter.pickle"
with open(pkl_cv, 'wb') as file:
    pickle.dump(cv, file)

# to test the data, uncomment below

# nb = "naive_bayes.pickle"
# with open(nb, 'rb') as file:
#     pickle_model = pickle.load(file)

## Testing the pickled model. IT WORKS!
# preds = pickle_model.predict(X_test_cv)
# print(preds)
# print("Just printed out the second set of predictions.")
# print(False in (preds == predictions))
