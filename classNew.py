import pandas as pd
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report , accuracy_score , confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn import model_selection


df = pd.read_csv('revNew01.csv')
print(df.head())


from io import StringIO

col = ['drugName', 'review']
df = df[col]
df = df[pd.notnull(df['review'])]

df.columns = ['drugName', 'review']

# # bersihkan dataset dari angka, tanda baca, dan spasi ganda
# # clean number
df.review = df.review.str.replace(r'\d+(\.\d+)?','')   

df.review = df.review.str.translate(str.maketrans('','',string.punctuation))


# # #clean punctuation
#prosesing = prosesing.str.replace(r'(^\m\d\s)','')

# # #clean white-space
df.review = df.review.str.replace(r'\s+',' ')
df.review = df.review.str.replace(r'^\s+|\s+?$',' ')


df['category_id'] = df['drugName'].factorize()[0]
#print(df['category_id'])

category_id_df = df[['drugName', 'category_id']].drop_duplicates().sort_values('category_id')
#print(category_id_df)

category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'drugName']].values)
#print(id_to_category)
df.head()

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(8,6))
# df.groupby('drugName').review.count().plot.bar(ylim=0)
# plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.review).toarray()
labels = df.category_id
print(labels)
print(features.shape)


from sklearn.feature_selection import chi2
import numpy as np

N = 2
for drugName, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(drugName))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['drugName'], random_state = 0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf)

clf = MultinomialNB().fit(X_train_tfidf, y_train)
#print(clf)


print(clf.predict(count_vect.transform(["This is my first time using any form of birth control. I&#039;m glad I went with the patch, I have been on it for 8 months. At first It decreased my libido but that subsided. The only downside is that it made my periods longer (5-6 days to be exact) I used to only have periods for 3-4 days max also made my cramps intense for the first two days of my period, I never had cramps before using birth control. Other than that in happy with the patch"])))

#df[df['review'] == "This is my first time using any form of birth control. I&#039;m glad I went with the patch, I have been on it for 8 months. At first It decreased my libido but that subsided. The only downside is that it made my periods longer (5-6 days to be exact) I used to only have periods for 3-4 days max also made my cramps intense for the first two days of my period, I never had cramps before using birth control. Other than that in happy with the patch"]

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    SVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
    DecisionTreeClassifier(),
    MLPClassifier(),
   
]

CV = 8
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  #print(model_name)

  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
  	entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

print(cv_df)



