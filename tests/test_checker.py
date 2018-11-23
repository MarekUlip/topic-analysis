import operator

class TestChecker:
    def __init__(self, training_docs, testing_docs, num_of_topics, log_writer):
        """

        :param testing_docs: docs in form of list containing tuples (topic_index, doc) used in test
        """
        self.testing_docs = testing_docs
        self.training_docs = training_docs
        self.num_of_topics = num_of_topics
        self.log_writer = log_writer
        self.representants = {}
        self.topic_indexes = {}
        self.topics_of_index = {}
        self.topic_distributions = []
        self.topic_numbers = []
        #count_of_filled_reps = 0
        self.prep_train_docs_for_assesment()
        self.count_topic_dist()
        self.confusion_matrix = [[0 for y in range(num_of_topics)] for x in range(num_of_topics)]

    def prep_train_docs_for_assesment(self, training_docs=None):
        if training_docs is not None:
            self.training_docs = training_docs
        for i in range(len(self.training_docs)):
            if self.training_docs[i][0] not in self.representants:
                self.representants[self.training_docs[i][0]] = [self.training_docs[i]]
            else:
                """if len(self.representants[self.training_docs[i][0]]) == max_rep_len:
                    continue"""
                self.representants[self.training_docs[i][0]].append(self.training_docs[i])
                """if len(self.representants[self.training_docs[i][0]]) == max_rep_len:
                    count_of_filled_reps += 1
                    if count_of_filled_reps == num_of_topics:
                        break"""

    def count_topic_dist(self):
        if len(self.representants) == 0:
            self.log_writer("Representants not set. Cannot make topic dist.")
            return
        for key, value in self.representants.items():
            self.topic_distributions.append(len(value)/len(self.training_docs))
            self.topic_numbers.append(key)

    def add_descriptions_to_confusion_matrix(self):
        for index, row in enumerate(self.confusion_matrix):
            row.insert(0,self.topic_numbers[index])
        topic_nums_for_matrix = self.topic_numbers.copy()
        topic_nums_for_matrix.insert(0,"")
        self.confusion_matrix.insert(0,topic_nums_for_matrix)


    def test_model(self, model, test_name):
        statistics = []
        stats = []
        for item in model.get_topics():
            statistics.append(item)
        statistics.append(["Article topic", "Model topic index"])
        self.connect_topic_id_to_topics(model)
        """for rep in self.representants:
            self.topic_positions[rep[0]] = max(model.analyse_text(rep[1]), key=lambda item: item[1])[0]"""
        """for i in range(self.num_of_topics):
            rep = self.representants[i+1].pop()
            self.topic_positions[rep[0]] = max(model.analyse_text(rep[1]), key=lambda item: item[1])[0]

        contains_duplicity = True
        dup_checker = {}
        while contains_duplicity:
            was_cycle_broken = False
            for key, value in self.topic_positions.items():
                if value not in dup_checker:
                    dup_checker[value] = 1
                else:
                    contains_duplicity = True
                    was_cycle_broken = True
                    dup_checker = {}
                    self.log_writer.add_log("Topic index estimation contains duplicity. Running estimation on new samples")
                    break
            if not was_cycle_broken:
                break
            for i in range(self.num_of_topics):
                if len(self.representants[i + 1]) == 0:
                    self.log_writer.add_log(
                        "Unable to establish correct index estimation skipping this test.")
                    return 0
                rep = self.representants[i + 1].pop()
                self.topic_positions[rep[0]] = max(model.analyse_text(rep[1]), key=lambda item: item[1])[0]"""

        for article in self.testing_docs:
            analysis_res = model.analyse_text(article[1])
            if len(analysis_res) == 0:
                print("nothing found")
                continue
            res = max(analysis_res, key=lambda item: item[1])
            statistics.append([article[0], res[0]])
            if res[0] not in self.topics_of_index:
                self.topics_of_index[res[0]] = [article[0]]
                self.topic_indexes[article[0]] = res[0]
                print("continuing")
                continue

            stats.append(1 if article[0] in self.topics_of_index[res[0]] else 0)
            topic_number_index = self.topic_numbers.index(article[0])

            if article[0] in self.topics_of_index[res[0]]:
                guessed_topic_number_index = self.topic_numbers.index(article[0])
            else:
                guessed_topic_number_index = self.topic_numbers.index(self.topics_of_index[res[0]][0])
            self.confusion_matrix[guessed_topic_number_index][topic_number_index] += 1
            #self.log_writer.add_log("Article with topic {} was assigned {} with {} certainty.".format(article[0], "correctly" if res[0] == self.topic_positions[article[0]] else "wrong", res[1]))

        self.log_writer.write_2D_list(test_name, statistics)
        self.add_descriptions_to_confusion_matrix()
        self.log_writer.write_2D_list(test_name+"-confusion-matrix", self.confusion_matrix)
        return sum(stats)/len(stats)


    def connect_topic_id_to_topics(self, model):
        #t = model.get_topics()
        for key, value in self.representants.items():
            connection_results = {}
            for article in value:
                #a = model.analyse_text(article[1])
                try:
                    topic_index = max(model.analyse_text(article[1]), key=lambda item: item[1])[0]
                except ValueError:
                    print("No topic index returned continuing")#TODO replace with if
                    continue
                if topic_index not in connection_results:
                    connection_results[topic_index] = 1
                else:
                    connection_results[topic_index] += 1

            best_candidates = max(connection_results.items(), key=operator.itemgetter(1))
            print(best_candidates)
            self.log_writer.add_log("Best candidate with index {} is connected to topic {} with {}% accuracy".format(best_candidates[0], key, (connection_results[best_candidates[0]]/len(value))*100))
            self.topic_indexes[key] = best_candidates[0]
            if best_candidates[0] not in self.topics_of_index:
                self.topics_of_index[best_candidates[0]] = [key]
            else:
                self.topics_of_index[best_candidates[0]].append(key)

        self.log_writer.add_log("Out of {} real topics only {} were learned".format(len(self.representants), len(self.topics_of_index)))



