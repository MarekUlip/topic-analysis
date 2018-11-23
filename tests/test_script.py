import os
import sys
import random

from methods.Lda import Lda
from methods.Lsa import Lsa
from methods.Hdp import Hdp

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from preprocessor.text_preprocessor import TextPreprocessor
from tests.log_writer import LogWriter
from tests.test_checker import TestChecker

def create_variations(depth, field, all_vars, possibilities):
    if depth == len(all_vars):
        possibilities.append(field)
        return

    for item in all_vars[depth]:
        f = [a for a in field]
        f.append(item)
        create_variations(depth + 1, f, all_vars, possibilities)

log_writer = LogWriter("log.txt")
base_path = os.getcwd()
csv_folder = base_path+"\\csv_folder\\"
data_sets = [(csv_folder+"1"+"\\train.csv",csv_folder+"1"+"\\test.csv",4),(csv_folder+"2"+"\\train.csv",csv_folder+"2"+"\\test.csv",14)]


strip_nums_params = use_stemmer_params = use_lemmatizer_params = strip_short_params = [True, False]
preproces_all_vals = [strip_nums_params, use_stemmer_params, use_lemmatizer_params, strip_short_params]
preproces_variations = []
create_variations(0, [], preproces_all_vals, preproces_variations)

lda_kappa = [0.51, 0.75, 1]
lda_tau = [0.0, 0.5, 1.0, 1.5, 2.0]
lda_minimum_probability = [0.0,0.01,0.1]
lda_passes = [20,50,100]
lda_iterations = [10,50,100]
lda_all_vals = [lda_kappa,lda_tau, lda_minimum_probability,lda_passes,lda_iterations]
lda_variations = []
create_variations(0,[],lda_all_vals,lda_variations)

lsa_one_pass = [True, False]
lsa_power_iter = [0, 1, 2, 5]
lsa_extra_samples = [1,100,500]
lsa_decay = [0.0,0.5,1.0,2.0]
lsa_all_vals = [lsa_one_pass, lsa_power_iter, lsa_extra_samples, lsa_decay]
lsa_variations = []
create_variations(0,[],lsa_all_vals,lsa_variations)

hdp_variations = []
num_of_test = 15

test_model = [False, True]

for i in range(len(data_sets)):
    statistics_to_merge = []
    for index, preproces_settings in enumerate(preproces_variations):
        seed = 5
        settings = {'strip_nums': preproces_settings[0],
            'use_stemmer': preproces_settings[1],
            'use_lemmatizer': preproces_settings[2],
            'strip_short': preproces_settings[3]
        }
        log_writer.add_log("Initializing text preprocessor with strip_nums: {}, use_stemmer: {}, use_lemmatizer {}, strip_short: {}.".format(preproces_settings[0], preproces_settings[1], preproces_settings[2], preproces_settings[3]))
        text_preprocessor = TextPreprocessor(settings)

        log_writer.add_log("Starting preprocessing texts for training")
        texts_for_train = text_preprocessor.load_and_prep_csv([data_sets[i][0]], "eng", False, 2, ',')[:100]
        log_writer.add_log("Preprocessing finished")

        log_writer.add_log("Starting preprocessing texts for testing")
        texts_for_testing = text_preprocessor.load_and_prep_csv([data_sets[i][1]], "eng", True, 2, ',')[:1000]
        log_writer.add_log("Preprocessing finished")

        lda_statistics = []
        # For every preprocesing add line that descripbes methods used
        lda_statistics.append(["strip_nums: {}, use_stemmer: {}, use_lemmatizer {}, strip_short: {}.".format(preproces_settings[0], preproces_settings[1], preproces_settings[2], preproces_settings[3])])
        for model_settings_index, model_settings in enumerate(lda_variations):
            # every column means one settings variation
            if model_settings_index == 0:
                lda_statistics.append([x for x in range(len(lda_variations))])
            for j in range(num_of_test):
                if model_settings_index == 0:
                    lda_statistics.append([])
                # every row means jth test
                test_checker_lda = TestChecker(texts_for_testing, data_sets[i][2], log_writer)
                lda = Lda(4, 15, kappa=model_settings[0], tau=model_settings[1], minimum_probability=model_settings[2], passes=model_settings[3], iterations=model_settings[4], random_state=seed)
                lda.train(texts_for_train)
                log_writer.add_log("Starting testing LDA model")
                accuracy = test_checker_lda.test_lda_model(lda, "\\results\\lda\\{}\\{}\\{}\\{}".format(i, model_settings_index, index, j))
                lda_statistics[j+2].append(accuracy)
                log_writer.add_log("Testing LDA model done with {}% accuracy".format(accuracy * 100))
                log_writer.add_log("\n\n")
        lda_statistics.append([])
        statistics_to_merge.append(lda_statistics)

        for model_settings_index, model_settings in enumerate(lsa_variations):
            for j in range(num_of_test):
                test_checker_lsa = TestChecker(texts_for_testing, data_sets[i][2], log_writer)
                lsa = Lsa(4, 15, one_pass=model_settings[0],power_iter=model_settings[1],extra_samples=model_settings[2],decay=model_settings[3])
                lsa.train(texts_for_train)
                log_writer.add_log("Starting testing LSA model")
                accuracy = test_checker_lsa.test_model(lsa, "\\results\\lsa\\{}\\{}\\{}\\{}".format(i, model_settings_index, index, j))
                log_writer.add_log("Testing LSA model done with {}% accuracy".format(accuracy * 100))
                log_writer.add_log("\n\n\n")

        """for model_settings_index, model_settings in enumerate(hdp_variations):
            for j in range(num_of_test):
                test_checker_hdp = TestChecker(texts_for_testing, data_sets[i][2], log_writer)
                hdp = Hdp(4, 15)
                hdp.train(texts_for_train)
                log_writer.add_log("Starting testing HDP model")
                accuracy = test_checker_hdp.test_model(hdp, "\\results\\hdp\\{}\\{}\\{}\\{}".format(i, model_settings_index, index, j))
                log_writer.add_log("Testing HDP model done with {}% accuracy".format(accuracy * 100))
                log_writer.add_log("\n\n\n")"""

    output_lda_csv = []
    for item in statistics_to_merge:
        for statistic in item:
            output_lda_csv.append(statistic)
    log_writer.write_statistics("\\results\\lda-summary\\LDAStats{}".format(i), output_lda_csv)

log_writer.end_logging()
