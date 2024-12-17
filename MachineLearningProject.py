# -*- coding: utf-8 -*-
"""MLProject.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FGCZXYYt6aAxWsuiSLm_OqbLyusekCmQ

Importing Libraries
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import missingno as msno #for missing values
from sklearn.impute import SimpleImputer , KNNImputer

#reading dataset
drive.mount('/content/gdrive')
data_dir = '/content/gdrive/My Drive/MLProject/'
df = pd.read_csv(f"{data_dir}/original_data.csv")
st= pd.read_csv(f"{data_dir}/score.csv")

# Commented out IPython magic to ensure Python compatibility.

# %matplotlib inline
sns.set(rc={'figure.figsize': [10, 10]}, font_scale=1.3)

df.sample(5)

df.describe()

msno.matrix(df, color=(99/255, 89/255, 133/255))
plt.show()

df.columns

for col in [ 'major', 'researchExp', 'industryExp', 'specialization',
       'toeflScore', 'program', 'department', 'toeflEssay', 'internExp',
       'greV', 'greQ', 'userProfileLink', 'journalPubs', 'greA', 'topperCgpa',
       'termAndYear', 'confPubs', 'ugCollege', 'gmatA', 'cgpa', 'gmatQ',
       'cgpaScale', 'gmatV']:
    print(f'-------{col}-------')
    print(df[col].unique())
    print('________________________'*3)

def sum_scor(scor):
    calc_scor = scor['researchExp']+scor['industryExp']+scor['toeflScore']+scor['internExp']+scor['greQ']+scor['greA']+scor['topperCgpa']+scor['cgpa']
    return calc_scor

df['TotalScore']=df.apply(sum_scor,axis=1)
df

df['season']=df['termAndYear'] .str.split('-').str[0]
df.head()

df['Year']=df['termAndYear'] .str.split('-').str[1]
df.head()

df['Year'] = pd.to_datetime(df['Year'], format='%Y/%m/%d', errors='coerce')
df['Year'] = df['Year'].dt.year
df['Year'].unique

imputer = KNNImputer()
df['Year'] = imputer.fit_transform(df[['Year']])
df['Year']=df['Year'].apply(int)
df['Year'].value_counts()

for col in [ 'researchExp','industryExp','toeflScore','internExp','greQ','greA','topperCgpa','cgpa','cgpaScale','toeflEssay','journalPubs','confPubs']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df[df['industryExp']==df['industryExp'].max()]

df[df['industryExp']==df['industryExp'].max()]

df[df['researchExp']==df['researchExp'].max()]

df.groupby("univName").describe()[['toeflScore','internExp','topperCgpa']].transpose()

df.groupby("univName").describe()[['topperCgpa','cgpa','cgpaScale']].transpose()

df[(df['userName']==1)][['admit']]

df[['userName']].value_counts()==1

df[(df['confPubs']==1)&(df['admit']==1)][['userName','ugCollege','major','univName']]

df.groupby("admit").describe()[['toeflScore','internExp','topperCgpa']].transpose()

sns.heatmap(df.corr()[['toeflScore','internExp','topperCgpa']],annot=True,cmap='coolwarm')

sns.heatmap(df.corr()[['greQ','greV','greA']],annot=True, cmap='viridis')

sns.set(style="white")
plt.rcParams['figure.figsize'] = (15, 10)
sns.heatmap(df.corr(), annot = True, linewidths=.5, cmap="Blues")
plt.title('Correlation Between Variables', fontsize = 30)
plt.show()

new_df=pd.pivot_table(data=df, values ='cgpa',columns ='univName',index ='admit')
new_df

new_df2=pd.pivot_table(data=df, values ='researchExp',columns ='univName',index ='admit')
new_df2

for new_df3 in [ 'industryExp','toeflScore','internExp','greQ','greA','topperCgpa','cgpa','cgpaScale']:
    print(f'-------{new_df3}-------')
    print(pd.pivot_table(data=df, values =[new_df3],columns ='univName',index ='admit'))
    print('________________________'*3)

[ 'researchExp','industryExp','toeflScore','internExp','greV','greQ','greA','topperCgpa','gmatA','cgpa','gmatQ','cgpaScale','','']

new_df2=pd.pivot_table(data=df, values ='toeflScore',columns ='univName',index ='admit')
new_df2

new_df2=pd.pivot_table(data=df, values ='internExp',columns ='univName',index ='admit')
new_df2

new_df2=pd.pivot_table(data=df, values ='greV',columns ='univName',index ='admit')
new_df2

new_df2=pd.pivot_table(data=df, values ='greQ',columns ='univName',index ='admit')
new_df2

new_df2=pd.pivot_table(data=df, values ='greA',columns ='univName',index ='admit')
new_df2

new_df2=pd.pivot_table(data=df, values ='cgpaScale',columns ='univName',index ='admit')
new_df2

new_df2=pd.pivot_table(data=df, values ='TotalScore',columns ='univName',index ='admit')
new_df2

df.groupby("admit").describe()['TotalScore'].transpose()

df[df['TotalScore']==df['TotalScore'].max()]

df[(df['admit']==1)&df['season']].mode()[0:1:1]

df[(df['admit']==1)&df['major']].mode()[0:1:1]

df[(df['admit']==1)&df['userName']].mode()[:2:1]

df[df['TotalScore']==df['TotalScore'].max()][:1:1][(df['admit']==1)&df['userName']].mode()

df[df['TotalScore']==df['TotalScore'].max()][:1:1][(df['admit']==1)]

df.groupby("univName").describe()['TotalScore'].transpose()

sns.displot(df['researchExp'],kde=False)

sns.displot(df['industryExp'],kde=False,color='m')

sns.displot(df['toeflScore'],kde=False,color='m')

sns.displot(df['internExp'],kde=False,color='m')

sns.displot(df['greV'],kde=False,color='m')

sns.displot(df['greQ'],kde=False,color='m')

sns.displot(df['topperCgpa'],kde=False, color=(5/255, 191/255, 202/255))

sns.displot(df['cgpa'],kde=False)

sns.displot(df['TotalScore'],kde=False)

df.columns

sns.kdeplot(y='confPubs',x='TotalScore',data=df,color=(179/255, 0/255, 94/255))

sns.kdeplot(y='confPubs',x='Year',data=df,color=(179/255, 0/255, 94/255))

sns.kdeplot(x='admit',y='Year',data=df,color=(179/255, 0/255, 94/255))

sns.kdeplot(y='admit',x='cgpa',data=df,color=(179/255, 0/255, 94/255))

sns.kdeplot(x='confPubs',y='topperCgpa',data=df,color=(179/255, 0/255, 94/255))

sns.boxplot(y='TotalScore',x='program',data=df,hue='admit',palette='pastel')

sns.boxplot(y='TotalScore',x='season',data=df,palette='pastel')

df.columns
df.isna().sum()/len(df)*100
df.drop(['gmatA','gmatQ','gmatV','userProfileLink','termAndYear','cgpaScale','journalPubs','toeflEssay','specialization'], axis=1, inplace=True)
df.isna().sum()/len(df)*100
df.columns
for col in ['userName', 'major', 'researchExp', 'industryExp', 'toeflScore',
       'program', 'department', 'internExp', 'greV', 'greQ', 'greA',
       'topperCgpa', 'confPubs', 'ugCollege', 'cgpa', 'univName', 'admit',
       'TotalScore', 'season', 'Year']:
    print(f'-------{col}-------')
    print(df[col].unique())
    print('________________________'*3)
numerical=['toeflScore','internExp','greV','greQ','greA','topperCgpa','TotalScore']
categoriacal=['major','department','ugCollege','season','program','confPubs']
from sklearn.impute import SimpleImputer , KNNImputer
imputer = KNNImputer()
for fit in numerical:
    df[[fit]] = imputer.fit_transform(df[[fit]])
df.isna().sum()/len(df)*100
imputer = SimpleImputer(strategy='most_frequent')
for fit in categoriacal:
    df[fit] = imputer.fit_transform(df[[fit]])
df.isna().sum()/len(df)*100
new_df=pd.pivot_table(data=df, values ='confPubs',columns ='major',index ='admit')
new_df=pd.pivot_table(data=df, values ='TotalScore',columns ='major',index ='admit')
df.groupby("univName").describe()[['topperCgpa','cgpa']].transpose()
df.groupby("major").describe()[['TotalScore','confPubs']].transpose()
df.groupby("program").describe()[['TotalScore','confPubs']].transpose()
df['confPubs'].unique()
df.drop_duplicates(subset='userName', keep='first',inplace= True)
df.drop(['TotalScore','userName'], axis=1, inplace=True)
numerical=['internExp','topperCgpa','greV','confPubs','toeflScore']
df = pd.get_dummies(df, columns=['major', 'program', 'department', 'ugCollege','univName','season'], drop_first=True)

import numpy as np
class LogRegression():
    def __init__(self, learning_rate=1, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X_Fold, Y_Fold):
        self.n_samples, self.n_features = X_Fold.shape
        self.weights = np.zeros(self.n_features)
        self.bias = 0

        for i in range(self.iterations):
            linear_model = np.dot(X_Fold, self.weights) + self.bias
            Log_Pred = self._sigmoid(linear_model)
            dw = (1 / self.n_samples) * np.dot(X_Fold.T, (Log_Pred - Y_Fold))
            db = (1 / self.n_samples) * np.sum(Log_Pred - Y_Fold)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X_Fold):
        linear_model = np.dot(X_Fold, self.weights) + self.bias
        Log_Pred = self._sigmoid(linear_model)
        Y_Pred = np.where(Log_Pred > 0.5, 1, 0)

        return Y_Pred

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, fbeta_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

X = df.drop('admit', axis=1)
y = df['admit']

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.2)
Scaler=StandardScaler()
Scaler.fit(X_Train)
X_Train = Scaler.transform(X_Train)
X_Test = Scaler.transform(X_Test)

MLModels = {
    "Naive Bayes": GaussianNB(),
    "LR": LogRegression(),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(),
    "DT": DecisionTreeClassifier( criterion='gini', min_samples_leaf=1,min_samples_split=3, min_weight_fraction_leaf=0.0,random_state=1, splitter='best'),
    "RF": RandomForestClassifier(criterion='gini'),
    "XGB": XGBClassifier()
}


def TrainModels(model, X_Train, Y_Train, X_Test, Y_Test):
    model.fit(X_Train, Y_Train)
    Y_Pred = model.predict(X_Test)
    TraningAccuracy = accuracy_score(Y_Train, model.predict(X_Train))
    TestingAccuracy=accuracy_score(Y_Test, Y_Pred)
    ConfusionMatrix = confusion_matrix(Y_Test, Y_Pred)
    Recall = recall_score(Y_Test, Y_Pred)
    Precision = precision_score(Y_Test, Y_Pred)
    F1Score = f1_score(Y_Test, Y_Pred)
    return TraningAccuracy,TestingAccuracy, ConfusionMatrix, Recall, Precision, F1Score

for name, model in MLModels.items():
    print(f'Training Model {name} \n')
    TraningAccuracy, TestingAccuracy, ConfusionMatrix, Recall, Precision, F1Score = TrainModels(model, X_Train, Y_Train, X_Test, Y_Test)
    TraningAccuracyPct = TraningAccuracy*100
    TestingAccuracyPct = TestingAccuracy*100
    RecallPct = Recall*100
    PrecisionPct = Precision*100
    F1ScorePct = F1Score*100
    print(f'Training Accuracy: {TraningAccuracyPct:.2f}%')
    print(f'Testing Accuracy: {TestingAccuracyPct:.2f}%')
    print(f'Confusion Matrix: \n{ConfusionMatrix}')
    print(f'Recall: {RecallPct:.2f}%')
    print(f'Precision: {PrecisionPct:.2f}%')
    print(f'F-1: {F1ScorePct:.2f}%')
    print('-'*30)

##### Ensembling voting Classifier #######
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
VotingML = {
    "LR": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(),
    "DT": DecisionTreeClassifier(criterion='gini', min_samples_leaf=1,min_samples_split=3, min_weight_fraction_leaf=0.0,random_state=1, splitter='best'),
    "RF": RandomForestClassifier(criterion='gini'),
    "XGB": XGBClassifier()
}
models = [(name, model) for name, model in VotingML.items()]
VotingModel = VotingClassifier(estimators=models, voting='hard', weights=[1,1,1,1,2,2])
VotingModel.fit(X_Train, Y_Train)
Voting_Pred = VotingModel.predict(X_Test)
VotingAccuracyTraining = accuracy_score(Y_Train, VotingModel.predict(X_Train))
VotingAccuracyTesting = accuracy_score(Y_Test,Voting_Pred)
print("Voting Classifier Training Accuracy: {:.2f}%".format(VotingAccuracyTraining*100))
print("Voting Classifier Testing Accuracy: {:.2f}%".format(VotingAccuracyTesting*100))
Voting_F1 = f1_score(Y_Test, Voting_Pred, average='weighted')
print("Voting Classifier F1 score: {:.2f}%".format(Voting_F1*100))