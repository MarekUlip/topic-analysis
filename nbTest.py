from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from preprocessor.text_preprocessor import TextPreprocessor
import numpy as np
import os
from sklearn.metrics import confusion_matrix

base_path = os.getcwd()
csv_folder = base_path+"\\csv_folder\\"
data_sets = [(csv_folder+"4",10,"-reuters-")]#, (csv_folder+"4",20,"-20newsgroups-")]#,(csv_folder+"1",10,"-reuters-")]#,(csv_folder+"2"+"\\train.csv",csv_folder+"2"+"\\test.csv",14)]
#data_sets = [(csv_folder+"2"+"\\train.csv",csv_folder+"2"+"\\test.csv",14)]


strip_nums_params = use_stemmer_params = use_lemmatizer_params = strip_short_params = [True, False]
preproces_all_vals = [strip_nums_params, use_stemmer_params, use_lemmatizer_params, strip_short_params]
#preproces_variations = []
preproces_variations = [[True,True,False,True],[True,True,True,False],[False,False,False,False]]

for index, preproces_settings in enumerate(preproces_variations):
    seed = 5
    settings = {'strip_nums': preproces_settings[0],
                'use_stemmer': preproces_settings[1],
                'use_lemmatizer': preproces_settings[2],
                'strip_short': preproces_settings[3]
                }
    text_preprocessor = TextPreprocessor(settings)

    """log_writer.add_log("Starting preprocessing texts of {} for training".format(data_sets[i][0]))
    texts_for_train = text_preprocessor.load_and_prep_csv([data_sets[i][0]+"\\train.csv"], "eng", False, 1, ';')
    log_writer.add_log("Preprocessing finished")"""

    texts_for_topic_asses = text_preprocessor.load_and_prep_csv([data_sets[0][0] + "\\train.csv"], "eng", True, 1, ';')
    texts_for_testing = text_preprocessor.load_and_prep_csv([data_sets[0][0] + "\\test.csv"], "eng", True, 1, ';')
    texts = texts_for_topic_asses.copy()
    texts.extend(texts_for_testing)
    articles = []
    topics = []
    for text in texts:
        articles.append(text[1])
        topics.append(text[0])

    counts = CountVectorizer().fit_transform(articles)
    tfidf = TfidfTransformer().fit_transform(counts)  # TODO try with fit_transform
    tst = tfidf[0]
    i = len(texts_for_topic_asses)
    """X_train, X_test, y_train, y_test = train_test_split(counts, topics, test_size=0.5, random_state=69)
    model = MultinomialNB().fit(X_train, y_train)
    predicted = model.predict(X_test)

    print(np.mean(predicted == y_test))
    print(confusion_matrix(y_test, predicted))"""
    model = MultinomialNB().fit(tfidf[0:i], topics[0:i])

    predicted = model.predict(tfidf[i:len(texts)])

    print(np.mean(predicted == topics[i:len(texts)]))
    print(confusion_matrix(topics[i:len(texts)], predicted))