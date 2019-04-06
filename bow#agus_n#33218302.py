# header program
# writter 
# nim  = 33218302
# nama = agus nursikuwagus
# prodi doktoral STEI
#------------------------------------------------------------
# dibuat sebagai salah tugas NLP
# tgl 4 April 2019
# menggunakan BOW
#------------------------------------------------------------

import pandas as pd
import numpy as np
import nltk
import string
from sklearn import model_selection 


# load dataset
filecsv = 'revNew01.csv'

# tampilkan pada modul pandas
df = pd.read_csv(filecsv, header = 0, delimiter = ',', encoding ='utf-8')

print(df)

# cek informasi dataset
print(df.info())
print(df.head())

# cek kelas target pada dataset 
targetclass = df['drugName']
print(targetclass.value_counts())

# # buat target kelas dengan multi-kelas
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()
Y = encoder.fit_transform(targetclass)
print(targetclass[:10])
#print(Y[:10])    # cek untuk 10 target sample

# # preprocessing untuk dataset 
teksreview = df['review']
#print(teksreview[:10])

# # bersihkan dataset dari angka, tanda baca, dan spasi ganda
# # clean number
prosesing = teksreview.str.replace(r'\d+(\.\d+)?','')   

prosesing = prosesing.str.translate(str.maketrans('','',string.punctuation))


# # #clean punctuation
#prosesing = prosesing.str.replace(r'(^\m\d\s)','')

# # #clean white-space
prosesing = prosesing.str.replace(r'\s+',' ')
prosesing = prosesing.str.replace(r'^\s+|\s+?$',' ')


# prosesing = prosesing.str.lower()
print(prosesing[:10])

# # hapus stopword menggunakan corpus english
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

prosesing = prosesing.apply(lambda x:' '.join(term for term in x.split() if term not in stop_words))

print(prosesing[:10])

# # proses stemming untuk mendapatkan hasil kata dasar
from nltk.stem.porter import *
stemmer = PorterStemmer()
prosesing = prosesing.apply(lambda x1:' '.join(stemmer.stem(term) for term in x1.split()))
print(prosesing[:10])

# **** buat vaktor kata dengan menggunakan bag-of-word ******
from nltk.tokenize import word_tokenize

pros_kata = []
for message in prosesing:
	kata = word_tokenize(message)
	for w in kata:
		pros_kata.append(w)

#print(pros_kata)
pros_kata = nltk.FreqDist(pros_kata)
print('Jumlah kata yang muncul {}'.format(len(pros_kata)))
print('Jumlah kata yang sering muncul {}'.format(pros_kata.most_common(15)))


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report , accuracy_score , confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['drugName'], test_size = 0.25, random_state = 7)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


clf  = MultinomialNB().fit(X_train_tfidf, y_train)
clf1 = KNeighborsClassifier(n_neighbors = 3).fit(X_train_tfidf, y_train)
clf2 = RandomForestClassifier(n_estimators=100, max_depth=2).fit(X_train_tfidf, y_train)
clf3 = LogisticRegression(solver='lbfgs',multi_class='multinomial').fit(X_train_tfidf, y_train)
clf4 = DecisionTreeClassifier().fit(X_train_tfidf, y_train)

clf6 = MLPClassifier().fit(X_train_tfidf, y_train)

#coba prediksi
predictions = clf.predict(count_vect.transform(["""My son is halfway through his fourth week of Intuniv. We became concerned when he began this last week, when he started taking the highest dose he will be on. For two days, he could hardly get out of bed, was very cranky, and slept for nearly 8 hours on a drive home from school vacation (very unusual for him.) I called his doctor on Monday morning and she said to stick it out a few days. See how he did at school, and with getting up in the morning. The last two days have been problem free. He is MUCH more agreeable than ever. He is less emotional (a good thing), less cranky. He is remembering all the things he should. Overall his behavior is better. 
We have tried many different medications and so far this is the most effective."""]))


predictions1 = clf1.predict(count_vect.transform(["""My son is halfway through his fourth week of Intuniv. We became concerned when he began this last week, when he started taking the highest dose he will be on. For two days, he could hardly get out of bed, was very cranky, and slept for nearly 8 hours on a drive home from school vacation (very unusual for him.) I called his doctor on Monday morning and she said to stick it out a few days. See how he did at school, and with getting up in the morning. The last two days have been problem free. He is MUCH more agreeable than ever. He is less emotional (a good thing), less cranky. He is remembering all the things he should. Overall his behavior is better. 
We have tried many different medications and so far this is the most effective."""]))


predictions2 = clf2.predict(count_vect.transform(["""My son is halfway through his fourth week of Intuniv. We became concerned when he began this last week, when he started taking the highest dose he will be on. For two days, he could hardly get out of bed, was very cranky, and slept for nearly 8 hours on a drive home from school vacation (very unusual for him.) I called his doctor on Monday morning and she said to stick it out a few days. See how he did at school, and with getting up in the morning. The last two days have been problem free. He is MUCH more agreeable than ever. He is less emotional (a good thing), less cranky. He is remembering all the things he should. Overall his behavior is better. 
We have tried many different medications and so far this is the most effective."""]))


predictions3 = clf3.predict(count_vect.transform(["""My son is halfway through his fourth week of Intuniv. We became concerned when he began this last week, when he started taking the highest dose he will be on. For two days, he could hardly get out of bed, was very cranky, and slept for nearly 8 hours on a drive home from school vacation (very unusual for him.) I called his doctor on Monday morning and she said to stick it out a few days. See how he did at school, and with getting up in the morning. The last two days have been problem free. He is MUCH more agreeable than ever. He is less emotional (a good thing), less cranky. He is remembering all the things he should. Overall his behavior is better. 
We have tried many different medications and so far this is the most effective."""]))

predictions4 = clf4.predict(count_vect.transform(["""My son is halfway through his fourth week of Intuniv. We became concerned when he began this last week, when he started taking the highest dose he will be on. For two days, he could hardly get out of bed, was very cranky, and slept for nearly 8 hours on a drive home from school vacation (very unusual for him.) I called his doctor on Monday morning and she said to stick it out a few days. See how he did at school, and with getting up in the morning. The last two days have been problem free. He is MUCH more agreeable than ever. He is less emotional (a good thing), less cranky. He is remembering all the things he should. Overall his behavior is better. 
We have tried many different medications and so far this is the most effective."""]))
print('DecisionTreeClassifier =', predictions4)


predictions6 = clf6.predict(count_vect.transform(["""My son is halfway through his fourth week of Intuniv. We became concerned when he began this last week, when he started taking the highest dose he will be on. For two days, he could hardly get out of bed, was very cranky, and slept for nearly 8 hours on a drive home from school vacation (very unusual for him.) I called his doctor on Monday morning and she said to stick it out a few days. See how he did at school, and with getting up in the morning. The last two days have been problem free. He is MUCH more agreeable than ever. He is less emotional (a good thing), less cranky. He is remembering all the things he should. Overall his behavior is better. 
We have tried many different medications and so far this is the most effective."""]))



print('MultinomialNB =', predictions)
print('KNeighborsClassifier = ',predictions1)
print('RandomForestClassifier', predictions2)
print('RandomForestClassifier', predictions3)
print('DecisionTreeClassifier =', predictions4)

print('MLPClassifier =',predictions6)



#hitung akurasi---------------------------------
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




#Akurasi MLPClassifier
print('MLPClassifier')
predictions6 = clf6.predict(X_train_tfidf)
print('Akurasi = ', accuracy_score(y_train, predictions6))
print('Matrix Confussion')
print(confusion_matrix(y_train, predictions6))
print(classification_report(y_train, predictions6))







