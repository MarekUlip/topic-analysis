class LSAModelTester:
    """
        class for testing Naive bayes model
    """
    def __init__(self, training_docs, testing_docs, num_of_topics, log_writer, topic_names):
        """
        :param training_docs: list of tuples in form of (topic ID, text)
        :param testing_docs: list of tuples in form of (topic ID, text)
        :param num_of_topics: Number of topics in dataset
        :param log_writer: Instance of LogWriter to write outputs
        :param topic_names: list of tuples in form of (topic ID, topic name)
        """
        self.testing_docs = testing_docs
        self.training_docs = training_docs
        self.num_of_topics = num_of_topics
        self.log_writer = log_writer
        self.topic_distributions = []
        self.topic_numbers = []
        self.topic_names = {}
        self.confusion_matrix = [[0 for y in range(num_of_topics)] for x in range(num_of_topics)]
        self.create_topic_names_dict(topic_names)

    def create_topic_names_dict(self, topic_names_list):
        """
        Creates dictionary where key is topic number and value is its name
        :param topic_names_list: list of tuples containing topic number[0] and its name[1]
        """
        for item in topic_names_list:
            self.topic_names[int(item[0])] = item[1]
            self.topic_numbers.append(int(item[0]))

    def add_descriptions_to_confusion_matrix(self):
        topic_names = []
        for topic_num in self.topic_numbers:
            topic_names.append(self.topic_names[topic_num])
        for index, row in enumerate(self.confusion_matrix):
            row.insert(0,topic_names[index])
        topic_names_for_matrix = topic_names.copy()
        topic_names_for_matrix.insert(0,"")
        self.confusion_matrix.insert(0,topic_names_for_matrix)

    def test_model(self, model, test_name):
        """
        Tests the precission of the provided model. Also writes out confuision matrix
        :param model: model to be tested
        :param test_name: name of the test
        :return: model accuracy in form of (accuracy %)/100. also outputs confusion matrix which is not part of return
        """
        stats = []
        for text in self.testing_docs:
            res = model.analyse_text(text[1])
            stats.append(1 if text[0] == res else 0)
            guessed_topic_number_index = self.topic_numbers.index(res)
            topic_number_index = self.topic_numbers.index(text[0])
            self.confusion_matrix[guessed_topic_number_index][topic_number_index] += 1
        self.add_descriptions_to_confusion_matrix()
        self.log_writer.write_2D_list(test_name+"-confusion-matrix", self.confusion_matrix)
        return sum(stats)/len(stats)