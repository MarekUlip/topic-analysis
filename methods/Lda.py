import os
import gensim
from gensim import corpora
from gensim.test.utils import datapath


class Lda:
    def __init__(self, topic_count, topic_word_count, alpha="auto", eta="auto", kappa=0.5, tau=1.0, minimum_probability=0.0, passes=20, iterations=5, random_state=5):
        self.topic_count = topic_count
        self.topic_word_count = topic_word_count
        self.alpha = alpha
        self.eta = eta
        self.kappa = kappa
        self.tau = tau
        self.minimum_probability = minimum_probability
        self.passes = passes
        self.iterations = iterations
        self.random_state = random_state
        self.dictionary = None
        self.model = None
        self.model_folder = os.getcwd()+"\\lda\\"
        self.model_path = self.model_folder+"model"
        self.dictionary_path = self.model_folder+"dictionary"

    def train(self, texts):
        self.dictionary = corpora.Dictionary(texts)
        # TODO maybe test work with dict self.dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=100000)
        doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in texts]
        self.model = gensim.models.LdaModel(
            doc_term_matrix,
            num_topics=self.topic_count,
            id2word=self.dictionary,
            alpha=self.alpha,
            # eta=int(self.eta.get()),
            decay=self.kappa,
            offset=self.tau,
            # random_state=5,
            passes=self.passes,
            iterations=self.iterations)

    def extract_important_words(self, topics, keep_values=True):
        d = {}
        i = 0
        for x in topics:
            a = x[1].replace(" ", "")
            a = a.replace("\"", "")
            d[i] = []
            for y in a.split("+"):
                if keep_values:
                    d[i].append(tuple(y.split("*")))
                else:
                    d[i].append(y.split("*")[1])
            i += 1
        return d

    def save_model(self):
        self.model.save(datapath(self.model_path))
        self.dictionary.save(datapath(self.dictionary_path))

    def load_model(self):
        self.model = gensim.models.LdaModel.load(self.model_path)
        self.dictionary = corpora.Dictionary.load(self.dictionary_path)

    def analyse_text(self, text):
        bow = self.dictionary.doc2bow(text.split())
        return self.model[bow]

    def get_topics(self):
        return self.model.print_topics(-1, self.topic_word_count)

