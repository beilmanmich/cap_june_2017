# TDI Capstone - June 2017

## Background & Motivation
As a final test of ability and growth, TDI requires a Capstone project upon completion of the 2 month bootcamp. This project sought to predict campaign funding success for [DonorsChoose.org](https://www.donorschoose.org/), a crowdfunding website that allows teachers to post campaigns (funding calls) for various educational purposes. Teachers submit details of their project along with a descriptive essay, this information populates a campaign webpage. Donors browse and select campaigns for donations, and if the project is successful in raising full funds, DonorsChoose purchases the materials and ships them to the teacher. The website is widely used - teachers at about two thirds of the public schools in the US have posted projects on the website, and it’s very effective - historically about 70% of campaigns receive full funding.

This leaves the question - what about the 30% that don’t receive full funding? Using predictive algorithms, can we build a data driven application to allow teachers to “proof” their campaign before fully launching with DonorsChoose? Will this predictive model apply the proverbial “red pen” suggesting edits/improvements to a teacher’s campaign essay? Can we go even further and match an essay from a previously successful campaign?

This project heavily utilized: _Python's sklearn, nltk, textblob, readcalc, pandas, flask_, and built off [prior data visualizations](https://github.com/beilmanmich/donors_dashboard).

**A quick example of the app interface, more details found on [YouTube](https://www.youtube.com/watch?v=DtTbJT360Vk)** [![IMAGE ALT TEXT HERE](https://github.com/beilmanmich/cap_june_2017/blob/master/gif/video.png)](https://www.youtube.com/watch?v=DtTbJT360Vk)

<div style="text-align:center"><img src ="https://github.com/beilmanmich/cap_june_2017/blob/master/gif/out.gif" /></div>

<div style="text-align:center"><img src ="https://github.com/beilmanmich/cap_june_2017/blob/master/gif/example.png" /></div>
<div style="text-align:center"><img src ="https://github.com/beilmanmich/cap_june_2017/blob/master/gif/example_nlp.png" /></div>

## About this Git
This Git contains the following folder structure:

### [_app_ folder: app files for local and heroku deploy](https://github.com/beilmanmich/cap_june_2017/tree/master/app)
[1. Final Heroku App](https://github.com/beilmanmich/cap_june_2017/blob/master/app/app.py)
[2. Necessary Static files (html, css, js, etc.)](https://github.com/beilmanmich/cap_june_2017/tree/master/app/static)
3. Format files ([requirements.txt](https://github.com/beilmanmich/cap_june_2017/blob/master/app/requirements.txt), [Procfile](https://github.com/beilmanmich/cap_june_2017/blob/master/app/Procfile), [dc_prediction.html](https://github.com/beilmanmich/cap_june_2017/blob/master/app/dc_prediction.html), [conda-requirements.txt](https://github.com/beilmanmich/cap_june_2017/blob/master/app/conda-requirements.txt) - used in conjunction with a Conda build pack

1 and 2 constitute all that is required to clone the prototype app for local development, upon download of this git users can simply run “app.py” on their local terminal, and launch the app via `localhost:5000`.

3, all other files (“format files”) indicate necessary files for Heroku (Flask) deployment (if deploying to Heroku a lean Anaconda build pack is recommended, examples in requirements). **A word of caution** this can be tricky/finicky to get right for deployment to a free [Heroku](https://devcenter.heroku.com/articles/getting-started-with-python#introduction) instance, as Heroku limits the final application footprint (slug) size to 300MB — [virtualenvs](http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/) allow for the development of a lean requirements.txt file and is strongly encouraged as best practice.

Pickled model and sample heroku psql model apps are included as previous versions, pickled models run well locally (and on an EC2 beanstalk), further memory increases can be realized by limiting data (more on this in data_clean).

### [_data_clean_ folder: Data Cleaning Code](https://github.com/beilmanmich/cap_june_2017/tree/master/data_clean)
This folder represents a simple data cleaning pipeline for merging data, creating feature variables, etc. This repo was built for local deployment, when deployed to Heroku I chose to upgrade size and row limits of psql db, which allows high functioning for short-term demo purposes. 

**File label numbering corresponds to pipeline order:**

[data_fetch.py](link) - may require edits (s3 links likely deprecated, DonorsChoose maintains documentation on it’s [OpenData page](https://research.donorschoose.org/t/download-opendata/33). All data licensed under creative commons, **CC NY 3.0**.

[1_clean_data.py](link)

[2_parse_recent.py](link) - when seeking to improve performance, reduce memory by parsing smaller date ranges

[3_dummy_data_munge.py](link)

[4_reduce_essays.py](link)

These files can be run from the terminal, print statements allow the user to track job progress for pipeline tasks. **Run these files before model_scripts**, these files munge data files for analysis, and, more importantly, parse data into usable file sizes (and formats, notably pkl). Edit these based on local and hosted memory requirements.

### [_model_scripts_ folder: Exploratory Data Analysis (EDA) Code](https://github.com/beilmanmich/cap_june_2017/tree/master/model_scripts)

In general, these scripts are a formal summary of several week’s of python notebook EDA (posts forthcoming), completed to inform model optimization. These models can be run directly in full from terminal, most codes contain print statements to compare cross-validated scores. Alternatively, one could copy/paste code segments into a ipython (jupyter) notebook for more “hands on” data exploratory analysis.

Again, one important distinction to note for this project is implementing a lean model to work in production. This said, the ultimate [DecisionTreeClassifier](https://en.wikipedia.org/wiki/Decision_tree_learning) used in the app is less accurate than many models contained in this repo. I ran [many models](https://github.com/beilmanmich/cap_june_2017/blob/master/model_scripts/ens_method.py), but ended up using a Decision Tree classifier to predict the success of project that a teacher posted on the site. I scored my models based on cross validated scores of their precision (examining confusion matrix), rationalizing that it was better to underestimate the probability of success on a project and have a teacher work a little harder to get funding rather than give them a false sense of security. My final model was able to correctly identify about 70-75% of the projects that ended up not getting funded.

This repository also contains natural language processing NLP codes, which explore a wide variety of techniques from Textblob, ReadCalc and nltk packages. [Tfidf transformations and weighting](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) was performed against n-grams of various size, ultimately selecting n-grams of 1 and 2 (mono and bigrams) based on cross validated model scores. [Most notable NLP readability models](https://github.com/beilmanmich/cap_june_2017/blob/master/model_scripts/calc_readability.py) explored include: [ARI](https://en.wikipedia.org/wiki/Automated_readability_index), [Coleman-Liau](https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index), [Flesch-Kincaid](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests), [Dale-Chall](https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula), [Gunning Fog](https://en.wikipedia.org/wiki/Gunning_fog_index), [LIX Index](https://en.wikipedia.org/wiki/LIX), and [SMOG grade](https://en.wikipedia.org/wiki/SMOG). Ultimately a polarity score and ARI were selected for scoring the teacher essay. Essay recommendations were rendered through [Textblob](https://textblob.readthedocs.io/en/dev/) based on key term matching (mono and bigrams).