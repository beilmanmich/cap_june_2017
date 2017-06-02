#MUNGING PROJECTS DATA FOR EDA AND MODEL FIT

from __future__ import division
import pandas as pd
import pickle
import numpy as np

pd.options.mode.chained_assignment = None

print "Hang in there, data munger extraordinaire...analysis awaits!"

df = pd.read_csv('../data/opendata_projects000.gz', escapechar='\\', names=['_projectid', '_teacher_acctid', '_schoolid', 'school_ncesid', 'school_latitude', 'school_longitude', 'school_city', 'school_state', 'school_zip', 'school_metro', 'school_district', 'school_county', 'school_charter', 'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise', 'teacher_prefix', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'primary_focus_subject', 'primary_focus_area' ,'secondary_focus_subject', 'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level', 'vendor_shipping_charges', 'sales_tax', 'payment_processing_charges', 'fulfillment_labor_materials', 'total_price_excluding_optional_support', 'total_price_including_optional_support', 'students_reached', 'total_donations', 'num_donors', 'eligible_double_your_impact_match', 'eligible_almost_home_match', 'funding_status','date_posted', 'date_completed', 'date_thank_you_packet_mailed', 'date_expiration'],
                 parse_dates=['date_posted', 'date_completed', 'date_thank_you_packet_mailed', 'date_expiration'])
print "Done reading csv"

print "Step1...creating live set"
# Pickle dump currently live projects as hold out set for final testing
live_df = df[df['funding_status'] == 'live']
with open('../data/live.pkl', 'w') as live_file:
    pickle.dump(live_df, live_file)
print "Step1...Done creating live set"

print "Step2a...adding response column..."
# Throw out reallocated and live projects and label rows
df2 = df[df['funding_status'] != "reallocated"]
df2 = df2[df2['funding_status'] != "live"]
df2['RESP'] = 0
df2['RESP'][df2['funding_status'] == 'completed'] = 1
per_remaining = len(df2) / len(df)
print "Percent of original data remaining: %0.2f" % (per_remaining * 100)
print "Step2a...Done adding response column"

print "Step2b...replacing binary features..."
# Replace binary features with 0,1
binary_features = ['school_charter',
                   'school_magnet',
                   'school_year_round',
                   'school_nlns',
                   'school_kipp',
                   'school_charter_ready_promise',
                   'teacher_teach_for_america',
                   'teacher_ny_teaching_fellow']
for feature in binary_features:
    df2[feature] = df2[feature].replace("t", 0)
    df2[feature] = df2[feature].replace("f", 1)
print "Step2b...Done replacing binary features"

print "Step3...adding date features..."
# Calculate month, quarter and year for date project was posted on site
df2['month'] = df2.date_posted.map(lambda x: x.month)
df2['quarter'] = df2.date_posted.map(lambda x: x.quarter)
df2['year'] = df2.date_posted.map(lambda x: x.year)
print "Step3...Done adding date features"

print "Step4...adding expiration dates and optional support features..."
# Feature engineering (expiration dates and amount/percentage of optional support)
df2['time_to_expire'] = df2.date_expiration - df2.date_posted
df2['amount_optional_support'] = df2.total_price_including_optional_support - df2.total_price_excluding_optional_support
df2['optional_support'] = 0
df2['optional_support'][df2['amount_optional_support'] > 0] = 1
df2['per_optional_support'] = df2.amount_optional_support / df2.total_price_excluding_optional_support
print "Step4...Done adding expiration dates and optional support features"

print "Step5...adding previous project features..."
# Feature engineering (previous projects posted by school & teacher)
school_posted = df2.set_index('date_posted').groupby('_schoolid').cumcount()
df2['school_previous_projects'] = school_posted.values
teacher_posted = df2.set_index('date_posted').groupby('_teacher_acctid').cumcount()
df2['teacher_previous_projects'] = teacher_posted.values
print "Step5...Done adding previous project features"

print "Step6...transforming continuous features"
# Feature engineering (Add 1 to continuous features that start at 0, log and sqrt some features)
df2.students_reached += 1
df2.total_price_excluding_optional_support += 1
df2.total_price_including_optional_support += 1
df2.vendor_shipping_charges += 1
df2.vendor_shipping_charges = df2.vendor_shipping_charges.replace(np.nan, df2.vendor_shipping_charges.mean())
df2['log_price_including'] = np.log(df2.total_price_including_optional_support)
df2['log_price_excluding'] = np.log(df2.total_price_excluding_optional_support)
df2['sqrt_students_reached'] = np.sqrt(df2.students_reached)
df2['log_vendor_shipping'] = np.log(df2.vendor_shipping_charges)
print "Step6...Done transforming continuous features"

print "Step7...adding teacher & gender features..."
# Using teacher prefix to get teacher gender
df2['teacher_gender'] = df2.teacher_prefix
df2.teacher_gender = df2.teacher_gender.replace("Mrs.", "Female")
df2.teacher_gender = df2.teacher_gender.replace("Ms.", "Female")
df2.teacher_gender = df2.teacher_gender.replace("Mr.", "Male")
df2.teacher_gender = df2.teacher_gender.replace("Dr.", np.nan)
df2.teacher_gender = df2.teacher_gender.replace("Mr. & Mrs.", np.nan)
# df2['teacher_gender_Female'] = df2.teacher_gender
# df2.teacher_gender_Female = df2.teacher_gender_Female.replace("Female",1)
# df2.teacher_gender_Female = df2.teacher_gender_Female.replace("Male",0)
# df2['teacher_gender_Male'] = df2.teacher_gender
# df2.teacher_gender_Male = df2.teacher_gender_Female.replace("Female",0)
# df2.teacher_gender_Male = df2.teacher_gender_Female.replace("Male",1)


print "Step7...Done adding teacher & gender features"

print "Step8...calculating project price per student..."
# Feature engineering (price of project per student reached)
df2['price_per_student'] = df2.total_price_including_optional_support / df2.students_reached
print "Step8...Done calculating project price per student"

print "Step9...creating bins..."
# Binning
df2['student_bins'] = pd.qcut(df2['students_reached'], 10, labels = False)
df2['price_in_bins'] = pd.qcut(df2['total_price_including_optional_support'], 10, labels = False)
df2['price_ex_bins'] = pd.qcut(df2['total_price_excluding_optional_support'], 10, labels = False)
df2['price_per_student_bins'] = pd.qcut(df2['price_per_student'], 10, labels = False)
print "Step9...Done creating bins"

print "Step10...adding state features..."
# Adding state features (average number of donors per project in each state)
df2.school_state = df2.school_state.replace("La", "LA")
state_df = pd.DataFrame(df2.num_donors.groupby([df2.school_state]).sum())
state_df['school_state'] = state_df.index
state_projects_df = pd.DataFrame(df2.num_donors.groupby([df2.school_state]).count())
state_projects_df['school_state'] = state_projects_df.index
state_df = state_df.merge(state_projects_df, on="school_state", how="left")
state_df.columns = ['total_state_donors', 'school_state', 'total_state_projects']
df2 = df2.merge(state_df, on="school_state", how="left")
df2['state_avg_donors'] = df2.total_state_donors / df2.total_state_projects
print "Step10...Done adding state features"

print "Step11...cleaning missing values & NaNs..."
# Dropping unnecessary columns and cleaning up missing values
df3 = df2[['_projectid',
           '_teacher_acctid',
           '_schoolid',
           'school_state',
           'school_metro',
           'school_charter',
           'school_magnet',
           'school_year_round',
           'school_nlns',
           'school_kipp',
           'school_charter_ready_promise',
           'teacher_teach_for_america',
           'teacher_ny_teaching_fellow',
           'primary_focus_subject',
           'primary_focus_area',
           'resource_type',
           'poverty_level',
           'grade_level',
           'vendor_shipping_charges',
           'total_price_excluding_optional_support',
           'total_price_including_optional_support',
           'students_reached',
           'date_posted',
           'RESP',
           'month',
           'quarter',
           'year',
           'time_to_expire',
           'optional_support',
           'school_previous_projects',
           'teacher_previous_projects',
           'log_price_including',
           'log_price_excluding',
           'sqrt_students_reached',
           'student_bins',
           'price_in_bins',
           'price_ex_bins',
           'teacher_gender',
           # 'teacher_gender_Female',
           # 'teacher_gender_Male',
           'price_per_student',
           'price_per_student_bins',
           'total_state_donors',
           'total_state_projects',
           'state_avg_donors',
           'per_optional_support']]
df3 = df3.dropna(
    subset=['school_metro', 'grade_level', 'students_reached', 'primary_focus_area', 'primary_focus_subject',
            'resource_type', 'time_to_expire', 'teacher_gender'])
print "Step11...Done cleaning missing values"

final_remaining = len(df3) / len(df2)
print "Percent of data still remaining: %0.2f" % (final_remaining * 100)

print "Pickling"
with open('../data/cleaned_data_with_features.pkl', 'wb') as picklefile:
    pickle.dump(df3, picklefile)
print "DONE"
