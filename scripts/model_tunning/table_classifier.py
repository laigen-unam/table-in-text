"""
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
        table_classifier.py  -fP '../../dataset/TablesPOS.txt' -fN '../DeNovo/ClassifiersFiles/NegativeClass.txt'


ARGUMENTS
        -fP, --filePositives: text file with sentences corresponding to positive class (tables) processed with
                              CoreNLP (Microbiology|Microbiology|NNP)
        -fN, --fileNegatives: text file with sentences corresponding to negative class (no tables) processed with
                              CoreNLP in the same way as the previous file
        -tS, --trainingSize [optional]: size of dataset to train the model. Default: 0.7

SEE ALSO
        Table_Identifier.py


"""

# LIBRARIES

import argparse

from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
import os
import stanza


#############################################################################################################

# COMMAND LINE ARGUMENT PASSING

parser = argparse.ArgumentParser(description="script that trains and saves classification models")

# path to text file with cases of positive class processed with CoreNLP
parser.add_argument(
  "-fP", "--filePositives",
  metavar="path/to/file/positiveClass",
  help="path to text file with POS of positive class",
  required=False)

# path to text file cases of negative class processed with CoreNLP
parser.add_argument(
  "-fN", "--fileNegatives",
  metavar="path/to/file/negativeClass",
  help="path to text file with POS of negative class",
  required=False)

parser.add_argument(
  "-tS", "--trainingSize",
  metavar="float",
  help="size of data for training (0.0 to 1.0)",
  required=False,
  default=0.7,
  type=float)

args = parser.parse_args()


#############################################################################################################

# VECTORIZATION AND DATA PREPROCESSING
# Directory and files to save trained model
# Storage directory of all required objects

savedir = 'model/'

filename_svm = os.path.join(savedir, 'ClassifierSVM.joblib')
filename_vec = os.path.join(savedir, 'Vectorizer.joblib')
filename_variance = os.path.join(savedir, 'Variance.joblib')

# Extract POS of cases files. Uncoment if the input files are the ones returned by CoreNLP preprocessing.
# POS_extraction()


# Vectorize features
print("\n###########################################################################################################")
print('\nVectorizing...')

tables = open("dataset/TablesPOS.txt").readlines()
normal_sentences = open("dataset/NoTablesPOS.txt").readlines()
whole_text = tables + normal_sentences

# all_sentences_tags = []
#  We use Stanza for tokenization and pos tagging.
# nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_no_ssplit=True)
# for test_file in whole_text:
#    pos_sentences = []
#    for sentence in test_file:
#        doc = nlp(sentence)
#        tokenized_tags = ([word.xpos for sent in doc.sentences for word in sent.words])
#        tokenized_text = ([word.text for sent in doc.sentences for word in sent.words])
#        pos_sentences.append(' '.join(tokenized_tags))
#    all_sentences_tags.append(pos_sentences)


# Asign a positive or negative value.
class_label = []
for x in range(len(whole_text)):
    if x < len(tables):
        class_label.append('TABLE')
    else:
        class_label.append('NONE TABLE')

print('Done!\n')


# Split data for train and test the model
print("\n\n##########################################################################################################")
print("\nTraining Classifier...")
X_train, X_test, y_train, y_test = train_test_split(whole_text, class_label, train_size=0.99, test_size=0.01)

# Vectorization 
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)
joblib.dump(vectorizer, filename_vec)


# Deletion of reapeted columns.
variance_filtter = VarianceThreshold()
variance_filtter.fit(X_train, y_train)
joblib.dump(variance_filtter, filename_variance)
X_train = variance_filtter.transform(X_train)
X_test = variance_filtter.transform(X_test)

#############################################################################################################

# TRAINING SVM CLASSIFIER

ClassifierSVM = SVC(C=23.0, gamma=0.41, shrinking=True, kernel='rbf', class_weight=None, probability=False)
ClassifierSVM.fit(X_train, y_train)
joblib.dump(ClassifierSVM, filename_svm)
print('Done!\n')


# Classification scores and metrics.
print("\n\n###########################################################################################################")
print("\nClassifier Scores:\n")
print("Precision (f1 Score):")
predict = ClassifierSVM.predict(X_test)
class_score = f1_score(y_test, predict, average=None)
print("Negative Class: %", class_score[0], "\nPositve Class: %", class_score[1])
print("\nConfusion Matrix: \n ", confusion_matrix(y_test, predict), '\n\n')
print(classification_report(y_test, predict))
print('\n\nAditional Info:')
print('Number of suport Vectors for each class: ', ClassifierSVM.n_support_)


# Path to the trained classifier for latter use.
print("\n\n###########################################################################################################")
print('\nPaths to model files:\n\n')
print(f"Classifier = joblib.load('{filename_svm}')")
print(f"Vectorizer = joblib.load('{filename_vec}')")
print(f"Variance Treshold = joblib.load('{filename_variance}')")
print('\n\n\n')
