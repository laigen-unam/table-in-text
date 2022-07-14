from PyPDF2 import PdfReader
import os
import argparse


parser = argparse.ArgumentParser(description='Path to directory were the articles in pdf format are.')
parser.add_argument(
    "-i", "--input-dir",
    metavar="path/to/input/dir",
    help="path to directory of pdf input files",
    required=False)
args = parser.parse_args()

all_test_files = os.listdir(args.input_dir)
for test_file in all_test_files:
    if not test_file.startswith('.'):
        with open(f"{args.input_dir}/{test_file}", 'r') as inFile:
            reader = PdfReader(f"{args.input_dir}/{test_file}")
            number_of_pages = len(reader.pages)
            page = reader.pages[0]
            text = page.extract_text()

        all_text = ''.join([reader.pages[i].extract_text() for i in range(number_of_pages)])
        text_sentences = all_text.split('.\n')

        with open(f"input_articles/in_txt/{test_file.split('.')[0]}.txt", '+w') as file:
            for sentence in text_sentences:
                towrite = sentence.replace('\n', ' ')
                file.write(towrite)
                file.write('\n')
