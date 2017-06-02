from __future__ import division
import pandas as pd
import pickle

pd.options.mode.chained_assignment = None

print "Step1...reading dataframe..."
df = pd.DataFrame(pickle.load(open('../data/cleaned_data_with_features.pkl', 'rb')))
print "Step1...Done reading dataframe"

print "Step2...pulling just projects since 2015..."
df2 = df[df['year'] >= 2016]
print "Step2...Done pulling just projects since 2014"

print "Step3...Pickling..."
with open('../data/clean_recent_data.pkl', 'w') as picklefile:
    pickle.dump(df2, picklefile)
print "Step3...Done Pickling"

print "DONE"
