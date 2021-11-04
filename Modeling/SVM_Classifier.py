'''
NAME
       SVM CLASSIFIER

VERSION
        1.0

AUTHOR
        Joel Rodriguez Herrera <joelrh@lcg.unam.mx>
        Axel Zagal Norman <dtorres@lcg.unam.mx>
        Dante Torres Adonis <azagal@lcg.unam.mx>

DESCRIPTION
        SVM RBF TABLE CLASSIFIER

CATEGORY
        NLP

USAGE
        python3 train_model.py -fP '../Dataset/PositiveClass.txt' -fN '../DeNovo/ClassifiersFiles/NegativeClass.txt'


ARGUMENTS
        -fP, --filePositives: text file with sentences corresponding to positive class (tables) processed with
                              CoreNLP (Microbiology|Microbiology|NNP)
        -fN, --fileNegatives: text file with sentences corresponding to negative class (no tables) processed with
                              CoreNLP in the same way as the previous file
        -tS, --trainingSize [optional]: size of dataset to train the model. Default: 0.7

SEE ALSO
        Table_Identifier.py


'''

    # LIBRARIES

import argparse
import scipy
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.feature_selection import SelectPercentile, SelectFpr, f_classif, chi2, mutual_info_classif
import os
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD



#############################################################################################################

    # FUNCTIONS

def POS_extraction():
    with open(args.fileNegatives, 'r') as file:
        file_sentences = file.read()
    with open(args.filePositives, 'r') as file:
        file_tables= file.read()
    with open('../NLP/POSNegativeClass.txt', '+w') as file:
        for sentence in file_sentences.split('\n'):
            for word in sentence.split(' '):
                pSpeech = word.split("|")
                if len(pSpeech) == 3:
                    file.write(f'{pSpeech[2]} ')
            file.write('\n')
    with open('../POSPositiveClass.txt', '+w') as file:
        for sentence in file_tables.split('\n'):
            for word in sentence.split(' '):
                pSpeech = word.split("|")
                if len(pSpeech) == 3:
                    file.write(f'{pSpeech[2]} ')
            file.write('\n')


#############################################################################################################


    # COMMAND LINE ARGUMENT PASSING

parser = argparse.ArgumentParser(description="script that trains and saves classification models")

## path to text file with cases of positive class processed with CoreNLP
parser.add_argument(
  "-fP", "--filePositives",
  metavar="path/to/file/positiveClass",
  help="path to text file with POS of positive class",
  required=True)

## path to text file cases of negative class processed with CoreNLP
parser.add_argument(
  "-fN", "--fileNegatives",
  metavar="path/to/file/negativeClass",
  help="path to text file with POS of negative class",
  required=True)

parser.add_argument(
  "-tS", "--trainingSize",
  metavar="float",
  help="size of data for training (0.0 to 1.0)",
  required=False,
  default = 0.7,
  type = float)

args = parser.parse_args()


#############################################################################################################


    # VECTORIZATION AND DATA PREPROCESSING

## Directory and files to save trained model
### Storage directory of all required objects

savedir = '../joblib/'

filename_svm = os.path.join(savedir, 'ClassifierSVM.joblib')
filename_vec = os.path.join(savedir, 'Vectorizer.joblib')
filename_variance = os.path.join(savedir, 'Variance.joblib')
filename_percentile = os.path.join(savedir, 'Percentile.joblib')

## Extract POS of cases files. Uncoment if the input files are the ones returned by CoreNLP preprocessing. 
#POS_extraction() 


## Vectorize features
print("\n#############################################################################################################")
print('\nVectorizing...')

whole_text = []
tables = open("../Dataset/TablesPOS.txt").readlines()
normal_sentences = open("../Dataset/NoTablesPOS.txt").readlines()
whole_text = tables + normal_sentences


## Asign a positive or negative value.
class_label = []
for x in range(len(whole_text)):
    if x < len(tables):
        class_label.append('SI')
    else:
        class_label.append('NO')

print('Done!\n')


## Split data for train and test the model
print("\n\n#############################################################################################################")
print("\nTraining Classifier...")
X_train, X_test, y_train, y_test = train_test_split(whole_text, class_label, train_size=0.75, test_size=0.85)

# Vectorization 
vectorizer = TfidfVectorizer(analyzer= 'char', ngram_range=(3,5))
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)
joblib.dump(vectorizer, filename_vec)


## Deletion of reapeted columns.
variance_filtter = VarianceThreshold()
variance_filtter.fit(X_train, y_train)
joblib.dump(variance_filtter, filename_variance)
X_train = variance_filtter.transform(X_train)
X_test = variance_filtter.transform(X_test) 



percentile_filtter = SelectPercentile(f_classif, percentile=50)
percentile_filtter.fit(X_train, y_train)
X_train = percentile_filtter.transform(X_train)
X_test = percentile_filtter.transform(X_test)
joblib.dump(percentile_filtter, filename_percentile)


#############################################################################################################


    # TRAINING SVM CLASSIFIER

ClassifierSVM =  SVC(C = 20, gamma= 0.308, shrinking=True, kernel='rbf', class_weight=None, probability=False)
ClassifierSVM.fit(X_train, y_train)
joblib.dump(ClassifierSVM, filename_svm)
print('Done!\n')


# Classification scores and metrics.
print("\n\n#############################################################################################################")
print("\nClassifier Scores:\n")
print("Precision (f1 Score):")
predict = ClassifierSVM.predict(X_test)
class_score = f1_score(y_test, predict, average=None)
print("Negative Class: %", class_score[0], "\nPositve Class: %", class_score[1])
print("\nConfusion Matrix: \n ",confusion_matrix(y_test, predict),'\n\n')
print(classification_report(y_test, predict))
print('\n\nAditional Info:')
print('Number of suport Vectors for each class: ', ClassifierSVM.n_support_)


    # ROC Curve

# Binarize the targets
y_ = label_binarize(y_train, classes=['SI', 'NO'])
y_test_ = label_binarize(y_test, classes=['SI', 'NO'])

y_score = ClassifierSVM.decision_function(X_test)

## Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, threshold = roc_curve(y_test_, y_score)
roc_auc = auc(fpr, tpr)


plt.title('ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.savefig(AUC-ROC.png')


    # Precision-Recall Curve

average_precision = average_precision_score(y_test_, y_score)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))
disp = plot_precision_recall_curve(ClassifierSVM, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
plt.savefig('../DeNovo/ClassifiersFiles/AUC-PR.png')

matrix = plot_confusion_matrix(ClassifierSVM, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for RBF SVM')
#plt.savefig('CM_RBF_SVM.png')


## Path to the trained classifier for latter use.
print("\n\n#############################################################################################################")
print('\nPaths to models:\n\n')
print(f"Classifier = joblib.load('{filename_svm}')")
print(f"Vectorizer = joblib.load('{filename_vec}')")
print(f"Variance Treshold = joblib.load('{filename_variance}')")
print(f"Percentile Best Features= joblib.load('{filename_percentile}')")
print('\n\n\n')
