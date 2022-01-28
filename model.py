import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from joblib import dump
from imblearn.under_sampling import RandomUnderSampler
data=pd.read_csv("fake_job_postings (1).csv")

data.drop(['job_id', 'salary_range', 'telecommuting', 'has_company_logo', 'has_questions'],axis=1,inplace = True)

data.fillna(' ', inplace=True)

#Create independent and Dependent Features
columns = data.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["fraudulent"]]
# Store the variable we are predicting 
target = "fraudulent"
# Define a random state 
state = np.random.RandomState(42)
X = data[columns]
Y = data["fraudulent"]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
under_sampler = RandomUnderSampler()
X_res, y_res = under_sampler.fit_resample(X, Y)

df1 = pd.DataFrame(X_res)
  
df3 = pd.DataFrame(y_res)
  
# the default behaviour is join='outer'
# inner join
  
result = pd.concat([df1, df3], axis=1, join='inner')
data=result 
def split(location):
    l = location.split(',')
    return l[0]
data['country'] = data.location.apply(split)


data['text'] = data['title']+' '+data['location']+' '+data['company_profile']+' '+data['description']+' '+data['requirements']+' '+data['benefits']+' '+data['industry']

X_train, X_test, y_train, y_test = train_test_split(data.text, data.fraudulent, test_size=0.01)
del data['title']
del data['location']
del data['department']
del data['company_profile']
del data['description']
del data['requirements']
del data['benefits']
del data['required_experience']
del data['required_education']
del data['industry']
del data['function']
del data['country']
del data['employment_type']
X_train, X_test, y_train, y_test = train_test_split(data.text, data.fraudulent, test_size=0.01)
vect = TfidfVectorizer(lowercase= True, max_features=1000, stop_words=ENGLISH_STOP_WORDS)
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
dt = DecisionTreeClassifier()
dt.fit(X_train_dtm, y_train)
pipeline = Pipeline(steps= [('tfidf', TfidfVectorizer(lowercase=True,
                                                      max_features=12000,
                                                      stop_words= ENGLISH_STOP_WORDS)),
                            ('model', DecisionTreeClassifier())])
pipeline.fit(X_train, y_train)
dump(pipeline, 'model.joblib')