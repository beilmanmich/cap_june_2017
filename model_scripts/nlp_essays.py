from __future__ import division
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
pd.options.mode.chained_assignment = None
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import cross_val_score, train_test_split

print "Step1...reading data..."
df = pd.read_csv("../app/static/merged_data.csv")
print "Step1...done reading data..."

print "Step2...throwing out floats..."
essay_df = df[['_projectid', 'RESP', 'essay']]
essay_df['new_essay'] = essay_df['essay'].map(lambda x: type(x))
essay_df = essay_df[essay_df.new_essay == str]
print "Step2...done throwing out floats"

print "percent remaining", len(essay_df)/len(df)

print "Step3...decoding data..."
essay_df.new_essay = essay_df['essay'].map(lambda x: x.decode('utf-8'))
print "Step3...done decoding"

print "Step4...transforming data..."
documents = essay_df.new_essay.tolist()
classes = essay_df.RESP.tolist()
print "Step4...done transforming data"

print "Step5...vectorizing data..."
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2))
doc_vectors = vectorizer.fit_transform(documents)
print "Step5...done vectorizing data...fitting model..." \
      ""
model = MultinomialNB().fit(doc_vectors, classes)
print "done fitting model"

precision = np.mean(cross_val_score(model, doc_vectors, classes, scoring='precision'))
cm = confusion_matrix(classes, model.predict(doc_vectors))
print "Precision", precision
print "Percentage off", cm[0][1]/(cm[0][0]+cm[0][1])
print cm