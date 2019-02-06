import os
import gensim
from gensim import corpora
from gensim.test.utils import datapath

class Lsa:
    def __init__(self, topic_count=5, topic_word_count=15, one_pass=True, power_iter=1, extra_samples=1, decay=1.0, use_tfidf=True, params=None):
        if params is not None:
            self.topic_count = params.get("topic_count", topic_count)
            self.topic_word_count = params.get("topic_word_count", topic_word_count)
            self.one_pass = params.get("one_pass", one_pass)
            self.power_iter = params.get("power_iter", power_iter)
            self.extra_samples = params.get("extra_samples", extra_samples)
            self.decay = params.get("decay", decay)
            self.use_tfidf = params.get("use_tfidf", use_tfidf)
        else:
            self.topic_count = topic_count
            self.topic_word_count = topic_word_count
            self.one_pass = one_pass
            self.power_iter = power_iter
            self.extra_samples = extra_samples
            self.decay = decay
            self.use_tfidf = use_tfidf
        self.dictionary = None
        self.model = None


    def train(self, texts):
        """
        Trains this model with provided texts
        :param texts: list of tuples in form (topic_id, text) topic ids dont matter here this format is used only because
        its more general - easier testing
        """
        texts = [text[1].split() for text in texts]
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