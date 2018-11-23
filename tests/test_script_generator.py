from preprocessor.text_preprocessor import TextPreprocessor
from tests.log_writer import LogWriter
from tests.test_checker import TestChecker
import os
import sys

base_body = ""

log_writer = LogWriter("tst")
initializers = [""]
methods = ["lda","lsa","tfidf","hdp"]
prep_options = ["lemmatize","stem","none", "both"]



train_csv = os.getcwd() + "\\test_file\\"
print(train_csv)
settings = {
            'strip_nums': False,
            'use_stemmer': False,
            'use_lemmatizer': True,
            'strip_short': False
        }

#text_preprocessor = TextPreprocessor()
#test_checker  = TestChecker(text_preprocessor.load_and_prep_csv([]))