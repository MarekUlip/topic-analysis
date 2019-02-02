import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import numpy as np

class LdaSklearn:
    def __init__(self, topic_count, passes=20, iterations=5):
        self.topic_count = topic_count
        self.passes = passes
        self.iterations = iterations
        self.model = None
        self.tf_vectorizer = TfidfVectorizer(max_features=25)
        self.model_folder = os.getcwd()+"\\lda\\"
        self.model_path = self.model_folder+"model"
        self.dictionary_path = self.model_folder+"dictionary"

    def train(self, texts):
        texts = [" ".join(word for word in text) for text in texts]
        train = self.tf_vectorizer.fit_transform(texts)
        self.model = LatentDirichletAllocation(n_components=self.topic_count, max_iter=self.iterations)
        self.model.n_iter_ = self.passes
        self.model.fit(train)

    def extract_important_words(self, topics, keep_values=True):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def analyse_text(self, text):
        test = self.tf_vectorizer.transform(["".join(word for word in text)])
        topic_dist = np.matrix(self.model.transform(test))
        return [(topic_dist.argmax(axis=1).item(0), 1)]

    def get_topics(self):
        return []#self.model.print_topics(-1, self.topic_word_count)

