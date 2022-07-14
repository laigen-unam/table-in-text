"""
NAME
       Table Identifier SVM

VERSION
        2.0

AUTHOR
        Axel Zagal Norman <azagal@lcg.unam.mx>
        Dante Torres Adonis <dtorres@lcg.unam.mx>
        Joel Rodriguez Herrera <joelrh@lcg.unam.mx>

DESCRIPTION
        Table identifier script

CATEGORY
        ML

USAGE
        python3 scripts/extract_tables.py -i examples/input_articles     

ARGUMENTS
        -iJ, --input_dir: Path to directory were the articles in txt format are.

SEE ALSO

"""

import os 
import stanza
import joblib
import argparse

# Defines comand line arguments
parser = argparse.ArgumentParser(description='Path to directory were the articles in txt format are.')
parser.add_argument(
    "-i", "--input-dir",
    metavar="path/to/input/dir",
    help="path to directory of text input files",
    required=False)
args = parser.parse_args()


# Lists of lists containing instances and tags/labels. 
num_tables = 0
files_no_tables = []
articles_tokens = []
article_pos_tags = []
all_test_files = os.listdir(args.input_dir)
# We use Stanza for tokenization and pos tagging.  
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_no_ssplit=True)

# Files containing different parameters of the previously trained model.
vectorizer = joblib.load('model/Vectorizer.joblib')
sel_variance = joblib.load('model/Variance.joblib')
classifier_rbf = joblib.load('model/ClassifierSVM.joblib')

print('Extracting tables...')
for test_file in all_test_files:
    if not test_file.startswith('.'):
        with open(f'{args.input_dir}/{test_file}', 'r') as inFile:
            text = inFile.readlines()
        pos_sentences = []
        text_sentences = []
        
        # Transform features of each file
        for sentence in text:
            doc = nlp(sentence)
            tokenized_tags = ([word.xpos for sent in doc.sentences for word in sent.words])
            tokenized_text = ([word.text for sent in doc.sentences for word in sent.words])
            pos_sentences.append(' '.join(tokenized_tags))
            text_sentences.append(' '.join(tokenized_text))  

        text_data = vectorizer.transform(pos_sentences)
        data_set = sel_variance.transform(text_data)
        tableidentifier_rbf = classifier_rbf.predict(data_set)

        # Only creates a result file if there is a table in the original file.
        thers_table = 0
        for answer in tableidentifier_rbf:
            if answer == 'TABLE':
                thers_table = 1

        # Saves tables in new files
        if thers_table == 1:
            with open(f"tables_found/{test_file.split('.')[0]}.txt", '+w') as file:
                for index in range(len(tableidentifier_rbf)):
                    if tableidentifier_rbf[index] == 'TABLE':
                        num_tables += 1
                        file.write(text_sentences[index])
                        file.write('\n\n')
        else:
            files_no_tables.append(test_file)

print('Done!\n\n\n')
print('Files containing tables wew saved on dir tables_found')
print('Number of tables found: ', num_tables)
print('Number of files wihtout tables: ', len(files_no_tables))
