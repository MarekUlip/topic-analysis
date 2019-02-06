import os
import sys

from methods.Lda import Lda
from methods.Lsa import Lsa
from methods.Hdp import Hdp

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from preprocessor.text_preprocessor import TextPreprocessor
from tests.log_writer import LogWriter
from tests.model_tester import ModelTester

log_writer = LogWriter("log.txt")


"""def create_variations(depth, field, all_vars, possibilities):
    if depth == len(all_vars):
        possibilities.append(field)
        return

    for item in all_vars[depth]:
        f = [a for a in field]
        f.append(item)
        create_variations(depth + 1, f, all_vars, possibilities)

strip_nums_params = use_stemmer_params = use_lemmatizer_params = strip_short_params = [True,False]
preproces_all_vals = [strip_nums_params, use_stemmer_params, use_lemmatizer_params, strip_short_params]
preproces_variations = []
create_variations(0, [], preproces_all_vals, preproces_variations)
print(preproces_variations)"""

### Header of testing script ###

# Test for dataset #
train_csv = os.getcwd() + "\\csv_folder\\1\\train.csv"
test_csv = os.getcwd() + "\\csv_folder\\1\\test.csv"

## Test for particular preproces settings ##
settings = {
            'strip_nums': True,
            'use_stemmer': False,
            'use_lemmatizer': False,
            'strip_short': True
        }

text_preprocessor = TextPreprocessor(settings)
log_writer.add_log("Starting preprocessing texts for training")
texts_for_train = text_preprocessor.load_and_prep_csv([test_csv],"eng",False,2,',')[:100]
log_writer.add_log("Preprocessing finished")

log_writer.add_log("Starting preprocessing texts for testing")
test_checker = ModelTester(text_preprocessor.load_and_prep_csv([test_csv], "eng", True, 2, ','), 4, log_writer)
log_writer.add_log("Preprocessing finished")


### Test for particular method ###
lda = Lda(4,15)
log_writer.add_log("Starting training LDA model")
lda.train(texts_for_train)
log_writer.add_log("Finished training LDA model")
log_writer.add_log("Starting testing LDA model")
accuracy = test_checker.test_model(lda,"\\test\\LDA_Test")
log_writer.add_log("Testing LDA model done with {}% accuracy".format(accuracy*100))
log_writer.add_log("\n\n\n")

"""lsa = Lsa(4,15)
log_writer.add_log("Starting training LSA model")
lsa.train(texts_for_train)
log_writer.add_log("Finished training LSA model")
log_writer.add_log("Starting testing LSA model")
accuracy = test_checker.test_model(lsa,"LSA_Test")
log_writer.add_log("Testing LSA model done with {}% accuracy".format(accuracy*100))
log_writer.add_log("\n\n\n")"""

"""hdp = Hdp(4,15)
hdp.train(texts_for_train)
log_writer.add_log("Starting training HDP model")
accuracy = test_checker.test_model(hdp,"HDP_Test")
log_writer.add_log("Finished training HDP model")
log_writer.add_log("Starting testing HDP model")
log_writer.add_log("Testing HDP model done with {}% accuracy".format(accuracy*100))
log_writer.add_log("\n\n\n")"""

log_writer.end_logging()


