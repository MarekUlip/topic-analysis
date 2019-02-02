import os
import sys
import random
import numpy
import time

from methods.Lda import Lda
from methods.Lsa import Lsa
from methods.Hdp import Hdp
from methods.Lda_sklearn import LdaSklearn

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

def get_time_in_millis():
    return int(round(time.time())*1000)


log_writer = LogWriter("log.txt")
base_path = os.getcwd()
csv_folder = base_path+"\\csv_folder\\"
data_sets = [(csv_folder+"1",10,"-reuters-")]#, (csv_folder+"4",20,"-20newsgroups-")]#,(csv_folder+"1",10,"-reuters-")]#,(csv_folder+"2"+"\\train.csv",csv_folder+"2"+"\\test.csv",14)]
#data_sets = [(csv_folder+"2"+"\\train.csv",csv_folder+"2"+"\\test.csv",14)]


strip_nums_params = use_stemmer_params = use_lemmatizer_params = strip_short_params = [True, False]
preproces_all_vals = [strip_nums_params, use_stemmer_params, use_lemmatizer_params, strip_short_params]
#preproces_variations = []
preproces_variations = [[False,False,False,False]]#[[True,True,True,True],[False,False,False,False],[True,False,True,False],[True,False,True,True]]
#create_variations(0, [], preproces_all_vals, preproces_variations)

lda_kappa = [0.51]
lda_tau = [2.0]
lda_minimum_probability = [0.0,0.01,0.1]
lda_passes = [50]
lda_iterations = [50]
lda_all_vals = [lda_kappa,lda_tau, lda_passes,lda_iterations]
#lda_variations = []
#create_variations(0,[],lda_all_vals,lda_variations)

lsa_one_pass = [False]
lsa_power_iter = [2]
lsa_use_tfidf = [True]
lsa_topic_nums = [data_sets[0][1]]
lsa_extra_samples = [100,200]
lsa_decay = [0.5,1.0,2.0]
lsa_all_vals = [lsa_one_pass, lsa_power_iter, lsa_use_tfidf,lsa_topic_nums]
#lsa_variations = []
#create_variations(0,[],lsa_all_vals,lsa_variations)

hdp_variations = []
num_of_test = 10

test_model = [True, True]
start_time = get_time_in_millis()

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

        log_writer.add_log("Starting preprocessing texts of {} for training".format(data_sets[i][0]))
        texts_for_train = text_preprocessor.load_and_prep_csv([data_sets[i][0]+"\\train.csv"], "eng", False, 1, ';')
        log_writer.add_log("Preprocessing finished")

        #log_writer.add_log("Starting preprocessing texts of {} for training".format(data_sets[i][0]))
        texts_for_topic_asses = text_preprocessor.load_and_prep_csv([data_sets[i][0]+"\\train.csv"], "eng", True, 1, ';')
        #log_writer.add_log("Preprocessing finished")

        log_writer.add_log("Starting preprocessing texts of {} for testing".format(data_sets[i][0]))
        texts_for_testing = text_preprocessor.load_and_prep_csv([data_sets[i][0]+"\\test.csv"], "eng", True, 1, ';')
        log_writer.add_log("Preprocessing finished")
        models_for_test = [Lda(data_sets[i][1], 15, kappa=lda_kappa[0], tau=lda_tau[0], passes=lda_passes[0], iterations=lda_iterations[0])]
                           #Lsa(data_sets[i][1], 15, one_pass=lsa_one_pass[0],power_iter=lsa_power_iter[0], use_tfidf=lsa_use_tfidf[0])]
#LdaSklearn(data_sets[i][1], passes=lda_passes[0],iterations=lda_iterations[0]),
        topic_names = text_preprocessor.load_csv([data_sets[i][0]+"\\topic-names.csv"])
        model_names = ["LDA"]
        statistics = []
        statistics.append(["{} strip_nums: {}, use_stemmer: {}, use_lemmatizer {}, strip_short: {}.".format(index,preproces_settings[0], preproces_settings[1], preproces_settings[2], preproces_settings[3])])

        for m_index, model in enumerate(models_for_test):
            if test_model[m_index]:
                # For every preprocesing add line that descripbes methods used
                accuracies = []
                statistics.append([])
                statistics.append([model_names[m_index]])
                statistics.append([x for x in range(num_of_test)])
                statistics[len(statistics)-1].append("Average")
                statistics.append([])
                for j in range(num_of_test):
                    # every row means jth test
                    test_checker_lda = TestChecker(texts_for_topic_asses, texts_for_testing, data_sets[i][1], log_writer, topic_names) #"""numpy.asfarray(test_checker_lda.topic_distributions)"""
                    log_writer.add_log("Starting training {} model".format(model_names[m_index]))
                    model.train(texts_for_train)
                    log_writer.add_log("Starting testing {} model".format(model_names[m_index]))
                    accuracy = test_checker_lda.test_model(model, "\\results\\results{}{}\\{}\\preprocess{}\\test_num{}".format(data_sets[i][2], start_time, model_names[m_index], index, j))
                    accuracies.append(accuracy)
                    statistics[len(statistics)-1].append(accuracy)
                    log_writer.add_log("Testing {} model done with {}% accuracy".format(model_names[m_index], accuracy * 100))
                    log_writer.add_log("\n\n")
                total_accuracy = sum(accuracies)/len(accuracies)
                log_writer.save_as_plot("\\results\\results{0}{1}\\{2}\\preprocess{3}\\graph{3}-{2}{0}{1}".format(data_sets[i][2], start_time, model_names[m_index], index), accuracies)
                statistics[len(statistics)-1].append(total_accuracy)
                log_writer.add_log("Total accuracy is: {}".format(total_accuracy))

        statistics.append([])
        statistics_to_merge.append(statistics)

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
    log_writer.write_2D_list("\\results\\results-stats\\stats{}{}".format(data_sets[i][2],start_time), output_lda_csv)

log_writer.end_logging()
