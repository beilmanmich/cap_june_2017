from __future__ import division
import pandas as pd
import pickle
from readcalc import readcalc

pd.options.mode.chained_assignment = None

print "Step1...reading csv..."
df = pd.read_csv("../app/static/merged_recent_data.csv")
print "Step1...done reading csv"

print "Step2...throwing out floats..."
essay_df = df[['_projectid', 'RESP', 'essay' ]]
essay_df['new_essay'] = essay_df['essay'].map(lambda x: type(x))
essay_df = essay_df[essay_df.new_essay == str]
print "Step2...done throwing out floats"


print "percent remaining", len(essay_df)/len(df)
print "Step3...decoding data..."
essay_df.new_essay = essay_df['essay'].map(lambda x: x.decode('utf-8'))
print "Step3...done decoding data"

print "Step4...running analysis...A(N)"


essay_df['ari'] = essay_df['new_essay'].map(lambda x: readcalc.ReadCalc(x).get_ari_index())
print "A1...ari index...done"
essay_df['coleman'] = essay_df['new_essay'].map(lambda x: readcalc.ReadCalc(x).get_coleman_liau_index())
print "A2...coleman liau index...done"
essay_df['flesch_grade'] = essay_df['new_essay'].map(lambda x: readcalc.ReadCalc(x).get_flesch_kincaid_grade_level())
print "A3...flesch kincaid grade level...done"
essay_df['flesch_ease'] = essay_df['new_essay'].map(lambda x: readcalc.ReadCalc(x).get_flesch_reading_ease())
print "A4...flesch reading ease score...done"
essay_df['dale'] = essay_df['new_essay'].map(lambda x: readcalc.ReadCalc(x).get_dale_chall_score())
print "A5...dale chall score...done"
essay_df['gunning'] = essay_df['new_essay'].map(lambda x: readcalc.ReadCalc(x).get_gunning_fog_index())
print "A6...gunning fog index...done"
essay_df['lix'] = essay_df['new_essay'].map(lambda x: readcalc.ReadCalc(x).get_lix_index())
print "A7...lix index...done"
essay_df['smog'] = essay_df['new_essay'].map(lambda x: readcalc.ReadCalc(x).get_smog_index())
print "A8...smog index...done"

print "Step4...running analysis...A1:8...done"

print "Pickling"
with open('../data/read_score.pkl', 'wb') as picklefile:
    pickle.dump(essay_df, picklefile)
print "DONE"
