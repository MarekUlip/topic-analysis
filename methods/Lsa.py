import os
import gensim
from gensim import corpora
from gensim.test.utils import datapath

class Lsa:
    def __init__(self, topic_count, topic_word_count, one_pass=True, power_iter=1, extra_samples=1, decay=1.0, use_tfidf=True):
        self.topic_count = topic_count
        self.topic_word_count = topic_word_count
        self.one_pass = one_pass
        self.power_iter = power_iter
        self.extra_samples = extra_samples
        self.decay = decay
        self.use_tfidf = use_tfidf
        self.dictionary = None
        self.model = None

        #topics = "\n".join((str(x[1])) for x in self.find_topics(get_texts(self.app.get_folder_name())))

    def train(self, texts):
        self.dictionary = corpora.Dictionary(texts)
        corpus = [self.dictionary.doc2bow(doc) for doc in texts]
        if self.use_tfidf:
            tf_idf = gensim.models.TfidfModel(corpus)
            corpus_tfidf = tf_idf[corpus]
        else:
            corpus_tfidf = corpus
        self.model = gensim.models.LsiModel(corpus_tfidf,
                         id2word=self.dictionary,
                         num_topics=self.topic_count,
                         onepass=self.one_pass,
                         power_iters=self.power_iter,
                         decay=self.decay)


        #return lsamodel.print_topics(num_topics=self.topic_count, num_words=self.topic_word_count)

    def analyse_text(self, text):
        bow = self.dictionary.doc2bow(text.split())
        return self.model[bow]

    def get_topics(self):
        return self.model.print_topics(-1, self.topic_word_count)