import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
 
# initialize dataframe and populate data
columnNames = ['PID','SCL','SCRamp','SCRfreq','HR','BVP','TEMP','ACC','IBI','RMSenergy','mfcc[1]','mfcc[2]','mfcc[3]','mfcc[4]','mfcc[5]','mfcc[6]','mfcc[7]','mfcc[8]','mfcc[9]','mfcc[10]','mfcc[11]','mfcc[12]','zcr','voiceProb','F0','pause_frequency','StateAnxiety','Language']
dataFrame = pd.read_csv('data.csv', header=None, names=columnNames, skiprows=1)
del dataFrame['PID']
# print(dataFrame)


y = dataFrame.Language

# feature list
features = ['RMSenergy', 'SCL', 'SCRamp', 'SCRfreq', 'TEMP', 'ACC', 'IBI', 'HR', 'BVP']

for x in range(1, len(features) + 1):
    print(features[0:x])
 
    XValues = dataFrame[features[0:x]]
 
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    # cv = KFold(n_splits=5)
 
    clf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
 
    scores = cross_val_score(clf, XValues, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print(scores)
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))