from sklearn import tree
import scipy
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, SelectFpr, f_classif, chi2, mutual_info_classif
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tempfile import mkdtemp
import joblib
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds


# Definicion de funciones

def classificator_score(clasificador, entrenamiento):
    print('Best score {}: {}\n'.format(myScorer, clasificador.best_score_))
    print('Best parameters:\n')
    best_parameters = clasificador.best_estimator_.get_params()
    for param in sorted(best_parameters.keys()):
        print("\t%s: %r\n" % (param, best_parameters[param]))
    predict = clasificador.predict(entrenamiento)
    print("Confusion Matrix: \n ",confusion_matrix(y_test, predict),'\n\n')
    return predict 
     

        
# Se crean los archivos vectorizados
print("***************************************** Casificaciones Tfdiv *******************************************\n\n\n")

## Vectorize features
print('\n************************************************************')
print('Vectorizing...')
posTables = open("../Dataset/TablesPOS.txt").readlines()
posSentenes = open("../Dataset/NoTablesPOS.txt").readlines()
posComplete = posTables + posSentenes


## Split data for train and test the model
class_label = []
for index in range(len(posComplete)):
    if index < len(posTables):
        class_label.append('SI')
    else:
        class_label.append('NO')


## Split data for train and test the model
print("\n\n#############################################################################################################")
print("\nTraining Classifier...")
X_train, X_test, y_train, y_test = train_test_split(posComplete, class_label, train_size=0.80, test_size=0.20)

# Vectorization 
vectorizer = TfidfVectorizer(analyzer= 'char', ngram_range=(3,5))
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

selector = VarianceThreshold()
selector = selector.fit(X_train,X_test)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

percentile_filtter = SelectPercentile(f_classif, percentile=50)
percentile_filtter.fit(X_train, y_train)
X_train = percentile_filtter.transform(X_train)
X_test = percentile_filtter.transform(X_test)


jobs = -1
paramGrid = []
crossV = 5
myScorer = make_scorer(f1_score, average = 'macro')

print("\n\n\n----------------------------------- Random Forest -----------------------------------------")


# Random forest
classifier = RandomForestClassifier()
algorithmName = "RandomForest"
paramGrid = {
    'n_estimators': [100, 150,200,300],
    'bootstrap': [True, False],
    'criterion': ["gini", "entropy"],
    'class_weight': ['balanced', None],
}

myClassifier = model_selection.RandomizedSearchCV(classifier, paramGrid,
                                            cv=crossV, n_jobs=jobs,
                                            scoring=myScorer)


myClassifier.fit(X_train, y_train)
predict = classificator_score(myClassifier, X_test)
print(classification_report(y_test, predict))


print("\n\n\n----------------------------------- SGDClassifier  -----------------------------------------")

classifier = SGDClassifier(loss = 'log')
algorithmName = "SGDClassifier"
paramGrid = {'alpha' : [10**(-x) for x in range(7)],
            'penalty' : ['elasticnet', 'l1', 'l2'],
            'l1_ratio' : [0.15, 0.25, 0.5, 0.75],
            'class_weight': ['balanced', None],}
        
myClassifier = model_selection.RandomizedSearchCV(classifier, paramGrid,cv=crossV,n_iter=100, n_jobs=jobs,scoring=myScorer)


myClassifier.fit(X_train, y_train)
predict = classificator_score(myClassifier, X_test)
print(classification_report(y_test, predict))



print("\n\n\n-------------------- Radial Basis Function Support Vector Machine RandomizedSearch --------------------")
classifier = SVC()
paramGrid = {'C': np.arange(1,70,0.5),
                         'gamma': np.arange(0,1,0.1),
                         'kernel': ['rbf','linear'], 'class_weight': ['balanced', None],}
RS = model_selection.RandomizedSearchCV(classifier, paramGrid, n_iter=200,cv=crossV, n_jobs=-1, scoring=myScorer, verbose = 1)
RS.fit(X_train, y_train)
predict = classificator_score(RS, X_test)
print(classification_report(y_test, predict))
