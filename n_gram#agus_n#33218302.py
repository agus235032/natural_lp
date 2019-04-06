# header program
# writter 
# nim  = 33218302
# nama = agus nursikuwagus
# prodi doktoral STEI
#------------------------------------------------------------
# dibuat sebagai salah tugas NLP
# tgl 4 April 2019
# menggunakan n-gram
#------------------------------------------------------------


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report , accuracy_score , confusion_matrix

# load dataset
filecsv = 'revNew01.csv'

# tampilkan pada modul pandas
df = pd.read_csv(filecsv, header = 0, delimiter = ',', encoding ='utf-8')
print(df.head())

from io import StringIO

col = ['drugName', 'review']
df = df[col]
df = df[pd.notnull(df['review'])]

df.columns = ['Product', 'cons_review']

df['category_id'] = df['Product'].factorize()[0]
category_id_df = df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Product']].values)
print(df.head())

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(8,6))
# df.groupby('drugName').Consumer_complaint_narrative.count().plot.bar(ylim=0)
# plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.cons_review).toarray()
labels = df.category_id
print(features.shape)


from sklearn.feature_selection import chi2
import numpy as np

N = 2
for Product, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(Product[:10]))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))




X_train, X_test, y_train, y_test = train_test_split(df['cons_review'], df['Product'], test_size = 0.25, random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



clf  = MultinomialNB().fit(X_train_tfidf, y_train)
clf1 = KNeighborsClassifier(n_neighbors = 2).fit(X_train_tfidf, y_train)
clf2 = RandomForestClassifier(n_estimators = 100, max_depth=2).fit(X_train_tfidf, y_train)
clf3 = LogisticRegression(solver='lbfgs',multi_class='multinomial').fit(X_train_tfidf, y_train)

# clf4 = SGDClassifier(max_iter = 100).fit(X_train_tfidf, y_train)
clf4 = SVC(gamma='scale', decision_function_shape='ovo').fit(X_train_tfidf, y_train)

clf6 = MLPClassifier().fit(X_train_tfidf, y_train)

predictions = clf.predict(count_vect.transform(["""My son is halfway through his fourth week of Intuniv. We became concerned when he began this last week, when he started taking the highest dose he will be on. For two days, he could hardly get out of bed, was very cranky, and slept for nearly 8 hours on a drive home from school vacation (very unusual for him.) I called his doctor on Monday morning and she said to stick it out a few days. See how he did at school, and with getting up in the morning. The last two days have been problem free. He is MUCH more agreeable than ever. He is less emotional (a good thing), less cranky. He is remembering all the things he should. Overall his behavior is better. 
We have tried many different medications and so far this is the most effective."""]))
print('MultinomialNB =', predictions)


predictions1 = clf1.predict(count_vect.transform(["""My son is halfway through his fourth week of Intuniv. We became concerned when he began this last week, when he started taking the highest dose he will be on. For two days, he could hardly get out of bed, was very cranky, and slept for nearly 8 hours on a drive home from school vacation (very unusual for him.) I called his doctor on Monday morning and she said to stick it out a few days. See how he did at school, and with getting up in the morning. The last two days have been problem free. He is MUCH more agreeable than ever. He is less emotional (a good thing), less cranky. He is remembering all the things he should. Overall his behavior is better. 
We have tried many different medications and so far this is the most effective."""]))
print('KNeighborsClassifier = ',predictions1)

predictions2 = clf2.predict(count_vect.transform(["""My son is halfway through his fourth week of Intuniv. We became concerned when he began this last week, when he started taking the highest dose he will be on. For two days, he could hardly get out of bed, was very cranky, and slept for nearly 8 hours on a drive home from school vacation (very unusual for him.) I called his doctor on Monday morning and she said to stick it out a few days. See how he did at school, and with getting up in the morning. The last two days have been problem free. He is MUCH more agreeable than ever. He is less emotional (a good thing), less cranky. He is remembering all the things he should. Overall his behavior is better. 
We have tried many different medications and so far this is the most effective."""]))
print('RandomForestClassifier', predictions2)

predictions3 = clf3.predict(count_vect.transform(["""My son is halfway through his fourth week of Intuniv. We became concerned when he began this last week, when he started taking the highest dose he will be on. For two days, he could hardly get out of bed, was very cranky, and slept for nearly 8 hours on a drive home from school vacation (very unusual for him.) I called his doctor on Monday morning and she said to stick it out a few days. See how he did at school, and with getting up in the morning. The last two days have been problem free. He is MUCH more agreeable than ever. He is less emotional (a good thing), less cranky. He is remembering all the things he should. Overall his behavior is better. 
We have tried many different medications and so far this is the most effective."""]))
print('LogisticRegression =', predictions3)

predictions4 = clf4.predict(count_vect.transform(["""My son is halfway through his fourth week of Intuniv. We became concerned when he began this last week, when he started taking the highest dose he will be on. For two days, he could hardly get out of bed, was very cranky, and slept for nearly 8 hours on a drive home from school vacation (very unusual for him.) I called his doctor on Monday morning and she said to stick it out a few days. See how he did at school, and with getting up in the morning. The last two days have been problem free. He is MUCH more agreeable than ever. He is less emotional (a good thing), less cranky. He is remembering all the things he should. Overall his behavior is better. 
We have tried many different medications and so far this is the most effective."""]))
print('SVM =', predictions4)

predictions6 = clf6.predict(count_vect.transform(["""My son is halfway through his fourth week of Intuniv. We became concerned when he began this last week, when he started taking the highest dose he will be on. For two days, he could hardly get out of bed, was very cranky, and slept for nearly 8 hours on a drive home from school vacation (very unusual for him.) I called his doctor on Monday morning and she said to stick it out a few days. See how he did at school, and with getting up in the morning. The last two days have been problem free. He is MUCH more agreeable than ever. He is less emotional (a good thing), less cranky. He is remembering all the things he should. Overall his behavior is better. 
We have tried many different medications and so far this is the most effective."""]))
print('MLPClassifier =',predictions6)


#Akurasi Multinomial
print('MultinomialNB')
predictions = clf.predict(X_train_tfidf)
print('Akurasi = ', accuracy_score(y_train, predictions))
print('Matrix Confussion')
print(confusion_matrix(y_train, predictions))
print(classification_report(y_train, predictions))


#Akurasi KNeighborsClassifier
print('KNeighborsClassifier')
predictions1 = clf1.predict(X_train_tfidf)
print('Akurasi = ', accuracy_score(y_train, predictions1))
print('Matrix Confussion')
print(confusion_matrix(y_train, predictions1))
print(classification_report(y_train, predictions1))


#Akurasi RandomForestClassifier
print('RandomForestClassifier')
predictions2 = clf2.predict(X_train_tfidf)
print('Akurasi = ', accuracy_score(y_train, predictions2))
print('Matrix Confussion')
print(confusion_matrix(y_train, predictions2))
print(classification_report(y_train, predictions2))


#Akurasi LogisticRegression
print('LogisticRegression')
predictions3 = clf3.predict(X_train_tfidf)
print('Akurasi = ', accuracy_score(y_train, predictions3))
print('Matrix Confussion')
print(confusion_matrix(y_train, predictions3))
print(classification_report(y_train, predictions3))



#Akurasi Support Vektor Machine
print('SVM')
predictions4 = clf4.predict(X_train_tfidf)
print('Akurasi = ', accuracy_score(y_train, predictions4))
print('Matrix Confussion')
print(confusion_matrix(y_train, predictions4))
print(classification_report(y_train, predictions4))


#Akurasi MLPClassifier
print('MLPClassifier')
predictions6 = clf6.predict(X_train_tfidf)
print('Akurasi = ', accuracy_score(y_train, predictions6))
print('Matrix Confussion')
print(confusion_matrix(y_train, predictions6))
print(classification_report(y_train, predictions6))

