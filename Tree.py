import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

import os
WD = 'C:\\git\\HuaweiHackaton\\'
os.chdir(WD)


#le.inverse_transform([leE.classes_])
#leD.inverse_transform([leD.classes_])

#Very Silly way of building a tree...just to test:
def getSuggestion(emotion, day, time):
    df = pd.read_csv('TreeDataSet.csv')
    
    dict_suggestions = df[['Vendor', 'Remarks']].set_index('Vendor').T.to_dict('Remarks')[0]
    #df.drop(['Remarks'], axis=1, inplace = True)

    leE = LabelEncoder()
    df['Emotion'] = leE.fit_transform(df['Emotion'])
    
    leD = LabelEncoder()
    df['DayofWeek'] =leD.fit_transform(df['DayofWeek'])

    leT = LabelEncoder()
    df['TimeofDay'] =leT.fit_transform(df['TimeofDay'])

    leV = LabelEncoder()
    df['Vendor'] = leV.fit_transform(df['Vendor'])
    
    
    model = tree.DecisionTreeClassifier()
    model.fit(df[['Emotion','DayofWeek','TimeofDay']], df['Vendor'])

    d={} 
    d['Emotion']   = leE.transform([emotion])[0]
    d['DayofWeek'] = leD.transform([day])[0] 
    d['TimeofDay'] = leT.transform([time])[0]
    
    NewRow = pd.DataFrame([d], columns=d.keys())

    pred = model.predict(NewRow)[0]    
    suggestion = leV.inverse_transform([pred])[0]
    
    return 'Why dont you go to {} (and \\ at) {}'.format(suggestion, dict_suggestions[suggestion])
