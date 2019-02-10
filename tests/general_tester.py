from tests.ModelType import ModelType
from tests.lda_lsa_model_tester import LModelTester
from tests.naive_bayes_model_tester import NBModelTester
from methods.Lda import Lda
from methods.Lsa import Lsa
from methods.Lda_sklearn import LdaSklearn
from methods.Naive_bayes import NaiveBayes
from methods.Hdp import Hdp



class GeneralTester:
    def __init__(self, log_writer, start_time):
        self.testing_docs = None
        self.training_docs = None
        self.num_of_topics = None
        self.log_writer = log_writer
        self.start_time = start_time
        self.topic_names = None
        self.model_results = []
        self.preprocess_style = ""
        self.preproces_results = {}
        self.num_of_tests = 1

    def set_new_dataset(self, num_of_topics, topic_names):
        self.num_of_topics = num_of_topics
        self.topic_names = topic_names

    def set_new_preprocess_docs(self, training_docs, testing_docs, preprocess_style):
        self.testing_docs = testing_docs
        self.training_docs = training_docs
        self.preprocess_style = preprocess_style

    def do_test(self, model_type, num_of_tests, statistics, params, test_params):
        """
        Do test on provided model type. Also sets things up before the test.
        :param model_type: ModelType enum for model that should be tested
        :param num_of_tests: number of tests to be performed on this model
        :param statistics: list to which accuracy and other information will be written
        :param params: Parameters for tested model
        :param test_params: Parameters for test
        """
        self.num_of_tests = num_of_tests
        accuracies = []
        statistics.append([])
        statistics.append(model_type.name)
        statistics.append([x for x in range(num_of_tests)])
        statistics[len(statistics) - 1].append("Average")
        statistics.append([])
        for i in range(num_of_tests):
            accuracy = self.test_model(model_type,
                                       self.create_test_name(test_params.get("dataset_name","none"), self.start_time, model_type.name, test_params.get("preprocess_index", 0), i),
                                       params)
            accuracies.append(accuracy)
            statistics[len(statistics) - 1].append(accuracy)
            self.log_writer.add_log("Testing {} model done with {}% accuracy".format(model_type, accuracy * 100))
            self.log_writer.add_log("\n\n")
        total_accuracy = sum(accuracies) / len(accuracies)
        self.log_writer.add_to_plot(model_type.name, accuracies)
        self.log_writer.draw_plot(model_type.name+" "+self.preprocess_style, "\\results\\results{0}{1}\\{2}\\preprocess{3}\\graph{3}-{2}{0}{1}".format(test_params.get("dataset_name","none"), self.start_time,
                                                                                      model_type.name, test_params.get("preprocess_index", 0)), num_of_tests)
        self.model_results.append((model_type.name, accuracies))
        if model_type in self.preproces_results:
            self.preproces_results[model_type].append((self.preprocess_style, accuracies))
        else:
            self.preproces_results[model_type] = [(self.preprocess_style, accuracies)]
        statistics[len(statistics) - 1].append(total_accuracy)
        self.log_writer.add_log("Total accuracy is: {}".format(total_accuracy))

    def test_model(self, model_type, test_name, params):
        """
        Runs actual test on a model
        :param model_type:  ModelType enum for model that should be tested
        :param test_name: name that will be used for creating output folder
        :param params: Parameters for tested model
        :return: Accuracy of provided model
        """
        model = None
        if model_type == ModelType.LDA:
            model = Lda(self.num_of_topics, params=params)
        elif model_type == ModelType.LSA:
            model = Lsa(self.num_of_topics, params=params)
        elif model_type == ModelType.LDA_Sklearn:
            model = LdaSklearn(self.num_of_topics, params=params)
        if model is not None:
            self.log_writer.add_log("Starting training {} model".format(model_type))
            model.train(self.training_docs)  # TODO watch out for rewrites
            self.log_writer.add_log("Starting testing {} model".format(model_type))
            tester = LModelTester(self.training_docs, self.testing_docs, self.num_of_topics, self.log_writer, self.topic_names)
            return tester.test_model(model,test_name)

        if model_type == ModelType.NB:
            model = NaiveBayes()
            self.log_writer.add_log("Starting training {} model".format(model_type))
            model.train(self.training_docs,self.testing_docs)
            self.log_writer.add_log("Starting testing {} model".format(model_type))
            tester = NBModelTester(self.training_docs, self.testing_docs, self.num_of_topics, self.log_writer, self.topic_names)
            return tester.test_model(model,test_name)

    def create_test_name(self, dataset_name, start_time, model_name, preprocess_index, test_num):
        return "\\results\\results{}{}\\{}\\preprocess{}\\test_num{}".format(dataset_name, start_time, model_name,
                                                                      preprocess_index, test_num)

    def output_model_comparison(self, dset_name):
        """
        Creates png chart that shows accuracy comparision of all tested model based on current preprocessing.
        """
        for result in self.model_results:
            self.log_writer.add_to_plot(result[0], result[1])
        self.log_writer.draw_plot("Porovnaní modelů", "\\results\\charts\\{}\\{}{}model-compare".format(self.start_time, self.preprocess_style[:2], dset_name), self.num_of_tests)
        self.model_results.clear()

    def output_preprocess_comparison(self, dset_name):
        """
        Creates png chart that shows accuracy impacts to model accuracy with different preprocessing settings
        :param dset_name: Name of actual dataset
        :return:
        """
        for key, value in self.preproces_results.items():
            for result in value:
                self.log_writer.add_to_plot(result[0], result[1])
            self.log_writer.draw_plot("Vliv preprocessingu na přesnost","\\results\\charts\\{}\\{}{}preprocess-compare".format(self.start_time, dset_name,key.name), self.num_of_tests)
        self.preproces_results.clear()
