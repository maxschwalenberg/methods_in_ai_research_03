import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from datacreator import datacreator


def create_bag_of_words(datacreator_instance: datacreator):
    vectorizer = CountVectorizer()

    bag = vectorizer.fit_transform(
        datacreator_instance.x_train + datacreator_instance.x_test
    )
