import os
import gensim
from gensim import corpora
from gensim.test.utils import datapath

class Hdp:
    def __init__(self, topic_count, topic_word_count):
        self.topic_count = topic_count
        self.topic_word_count = topic_word_count
        self.model = None
        self.dictionary = None

    """def start(self):
        topics = "\n".join((str(x[1])) for x in self.find_topics(get_texts(self.app.get_folder_name())))"""

    def train(self, texts):
        self.dictionary = corpora.Dictionary(texts)
        doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in texts]
        self.model = gensim.models.HdpModel(doc_term_matrix, id2word=self.dictionary)
        #return hdpmodel.print_topics(num_topics=int(self.topic_count.get()), num_words=int(self.topic_word_count.get()))

    def analyse_text(self, text):
        bow = self.dictionary.doc2bow(text.split())
        return self.model[bow]

    def get_topics(self):
        return self.model.print_topics(-1, self.topic_word_count)
