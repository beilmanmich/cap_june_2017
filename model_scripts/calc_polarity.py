from __future__ import division
import pandas as pd
import pickle
from textblob import TextBlob

pd.options.mode.chained_assignment = None

print "Step1...reading csv"
df = pd.read_csv("../app/static/merged_recent_data.csv")
print "Step1...done reading csv"

print "Step2...parsing floats..."
essay_df = df[['_projectid', 'RESP', 'essay' ]]
essay_df['new_essay'] = essay_df['essay'].map(lambda x: type(x))
print "Step2...done parsing floats"

print "Step3...throwing out floats..."
essay_df = essay_df[essay_df.new_essay == str]
print "Step3...done throwing out floats"

print "Step4...decoding data..."
print "percent remaining", len(essay_df)/len(df)
essay_df.new_essay = essay_df['essay'].map(lambda x: x.decode('utf-8'))
print "Step4...done decoding"

print "Step5...assiging polar subjectivity..."
essay_df['polar'] = essay_df.new_essay.map(lambda x: TextBlob(x).polarity)
essay_df['subjectivity'] = essay_df.new_essay.map(lambda x: TextBlob(x).subjectivity)
print "Step5...done assigning polar subjectivity"

print "Pickling"
with open('../data/polar_score.pkl', 'wb') as picklefile:
    pickle.dump(essay_df, picklefile)
print "DONE"
