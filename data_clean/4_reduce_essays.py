from __future__ import division
import pandas as pd
import pickle

pd.options.mode.chained_assignment = None

print "Step1...reading essay csv..."
# df = pd.read_csv("../data/cleaned_essays.csv", names=['_projectid', '_teacherid', 'title', 'short_description', 'need_statement', 'essay', 'thankyou_note', 'impact_letter'])
df = pd.read_csv('../data/opendata_essays000.gz', escapechar='\\', names=['_projectid', '_teacherid', 'title', 'short_description', 'need_statement', 'essay', 'thankyou_note', 'impact_letter'])
print "Step1...Done reading essay csv"

print "Step2...reading recent data file..."
recent_df = pd.DataFrame(pickle.load(open('../data/dummied_recent_data.pkl', 'rb')))

merged_recent = recent_df.merge(df, how="left", on='_projectid')
merged_recent = merged_recent[[
    '_projectid',
    'school_previous_projects',
    'teacher_previous_projects',
    'month',
    'quarter',
    'year',
    'log_price_including',
    'sqrt_students_reached',
    'price_per_student',
    'total_state_donors',
    'total_state_projects',
    'state_avg_donors',
    'primary_focus_subject_Applied Sciences',
    'primary_focus_subject_Character Education',
    'primary_focus_subject_Civics & Government',
    'primary_focus_subject_College & Career Prep',
    'primary_focus_subject_Community Service',
    'primary_focus_subject_ESL',
    'primary_focus_subject_Early Development',
    'primary_focus_subject_Economics',
    'primary_focus_subject_Environmental Science',
    'primary_focus_subject_Extracurricular',
    'primary_focus_subject_Financial Literacy',
    'primary_focus_subject_Foreign Languages',
    'primary_focus_subject_Gym & Fitness',
    'primary_focus_subject_Health & Life Science',
    'primary_focus_subject_Health & Wellness',
    'primary_focus_subject_History & Geography',
    'primary_focus_subject_Literacy',
    'primary_focus_subject_Literature & Writing',
    'primary_focus_subject_Mathematics',
    'primary_focus_subject_Music',
    'primary_focus_subject_Nutrition',
    'primary_focus_subject_Other',
    'primary_focus_subject_Parent Involvement',
    'primary_focus_subject_Performing Arts',
    'primary_focus_subject_Social Sciences',
    'primary_focus_subject_Special Needs',
    'primary_focus_subject_Team Sports',
    'primary_focus_subject_Visual Arts',
    'poverty_level_high poverty',
    'poverty_level_highest poverty',
    'poverty_level_low poverty',
    'poverty_level_moderate poverty',
    'grade_level_Grades 3-5',
    'grade_level_Grades 6-8',
    'grade_level_Grades 9-12',
    'grade_level_Grades PreK-2',
    'vendor_shipping_charges',
    'school_metro_rural',
    'school_metro_suburban',
    'school_metro_urban',
    'teacher_gender_Female',
    'teacher_gender_Male',
    'resource_type_Books',
    'resource_type_Other',
    'resource_type_Supplies',
    'resource_type_Technology',
    'resource_type_Trips',
    'resource_type_Visitors',
    'per_optional_support',
    'optional_support',
    'teacher_teach_for_america',
    'RESP',
    'essay']]
print "Step2...Done reading recent data file"

# ['_projectid',
#                  'school_previous_projects',
#                  'teacher_previous_projects',
#                  'month',
#                  'log_price_including',
#                  'sqrt_students_reached',
#                  'price_per_student',
#                  'total_state_donors',
#                  'total_state_projects',
#                  'state_avg_donors',
#                  'primary_focus_subject_Applied Sciences',
#                  'primary_focus_subject_Character Education',
#                  'primary_focus_subject_Civics & Government',
#                  'primary_focus_subject_College & Career Prep',
#                  'primary_focus_subject_Community Service',
#                  'primary_focus_subject_ESL',
#                  'primary_focus_subject_Early Development',
#                  'primary_focus_subject_Economics',
#                  'primary_focus_subject_Environmental Science',
#                  'primary_focus_subject_Extracurricular',
#                  'primary_focus_subject_Financial Literacy',
#                  'primary_focus_subject_Foreign Languages',
#                  'primary_focus_subject_Gym & Fitness',
#                  'primary_focus_subject_Health & Life Science',
#                  'primary_focus_subject_Health & Wellness',
#                  'primary_focus_subject_History & Geography',
#                  'primary_focus_subject_Literacy',
#                  'primary_focus_subject_Literature & Writing',
#                  'primary_focus_subject_Mathematics',
#                  'primary_focus_subject_Music',
#                  'primary_focus_subject_Nutrition',
#                  'primary_focus_subject_Other',
#                  'primary_focus_subject_Parent Involvement',
#                  'primary_focus_subject_Performing Arts',
#                  'primary_focus_subject_Social Sciences',
#                  'primary_focus_subject_Special Needs',
#                  'primary_focus_subject_Team Sports',
#                  'primary_focus_subject_Visual Arts',
#                  'poverty_level_high poverty',
#                  'poverty_level_highest poverty',
#                  'poverty_level_low poverty',
#                  'poverty_level_moderate poverty',
#                  'grade_level_Grades 3-5',
#                  'grade_level_Grades 6-8',
#                  'grade_level_Grades 9-12',
#                  'grade_level_Grades PreK-2',
#                  'school_metro_rural',
#                  'school_metro_suburban',
#                  'school_metro_urban',
#                  'resource_type_Books',
#                  'resource_type_Other',
#                  'resource_type_Supplies',
#                  'resource_type_Technology',
#                  'resource_type_Trips',
#                  'resource_type_Visitors',
#                  'RESP',
#                  'essay']

print "Step3...reading full data file..."
full_df = pd.DataFrame(pickle.load(open('../data/dummied_full_data.pkl', 'rb')))

merged_full = full_df.merge(df, how="left", on='_projectid')
merged_full = merged_full[[
    '_projectid',
    'school_previous_projects',
    'teacher_previous_projects',
    'month',
    'quarter',
    'year',
    'log_price_including',
    'sqrt_students_reached',
    'price_per_student',
    'total_state_donors',
    'total_state_projects',
    'state_avg_donors',
    'primary_focus_subject_Applied Sciences',
    'primary_focus_subject_Character Education',
    'primary_focus_subject_Civics & Government',
    'primary_focus_subject_College & Career Prep',
    'primary_focus_subject_Community Service',
    'primary_focus_subject_ESL',
    'primary_focus_subject_Early Development',
    'primary_focus_subject_Economics',
    'primary_focus_subject_Environmental Science',
    'primary_focus_subject_Extracurricular',
    'primary_focus_subject_Financial Literacy',
    'primary_focus_subject_Foreign Languages',
    'primary_focus_subject_Gym & Fitness',
    'primary_focus_subject_Health & Life Science',
    'primary_focus_subject_Health & Wellness',
    'primary_focus_subject_History & Geography',
    'primary_focus_subject_Literacy',
    'primary_focus_subject_Literature & Writing',
    'primary_focus_subject_Mathematics',
    'primary_focus_subject_Music',
    'primary_focus_subject_Nutrition',
    'primary_focus_subject_Other',
    'primary_focus_subject_Parent Involvement',
    'primary_focus_subject_Performing Arts',
    'primary_focus_subject_Social Sciences',
    'primary_focus_subject_Special Needs',
    'primary_focus_subject_Team Sports',
    'primary_focus_subject_Visual Arts',
    'poverty_level_high poverty',
    'poverty_level_highest poverty',
    'poverty_level_low poverty',
    'poverty_level_moderate poverty',
    'grade_level_Grades 3-5',
    'grade_level_Grades 6-8',
    'grade_level_Grades 9-12',
    'grade_level_Grades PreK-2',
    'vendor_shipping_charges',
    'school_metro_rural',
    'school_metro_suburban',
    'school_metro_urban',
    'teacher_gender_Female',
    'teacher_gender_Male',
    'resource_type_Books',
    'resource_type_Other',
    'resource_type_Supplies',
    'resource_type_Technology',
    'resource_type_Trips',
    'resource_type_Visitors',
    'per_optional_support',
    'optional_support',
    'teacher_teach_for_america',
    'RESP',
    'essay']]
print "Step3...Done reading full data file..."

print "Step4...writing recent data to csv in app folder..."
merged_recent.to_csv("../app/static/merged_recent_data.csv", index=False)
print "Step4...Done writing recent data to csv in app folder"

print "Step5...writing full data to csv in app folder..."
merged_full.to_csv("../app/static/merged_full_data.csv", index=False)
print "Step5...Done writing full data to csv in app folder"

print "DONE"
