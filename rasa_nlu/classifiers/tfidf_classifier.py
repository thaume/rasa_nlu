from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import typing
from builtins import zip
import os
import io
from future.utils import PY3
from typing import Any, Optional
from typing import Dict
from typing import List
from typing import Text
from typing import Tuple
from textblob import TextBlob as tb
import math

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData


class TFIDFClassifier(Component):

    name = "tfidf_classifier"

    provides = ["intent"]

    def __init__(self, model=None, le=None):
        """Construct a new intent classifier using the keras framework."""
        from sklearn.preprocessing import LabelEncoder

        self.MAX_DOC_LENGTH = 20
        self.EMBEDDING_DIM = 50
        self.TFIDF_THRESHOLD = 0.20
        self.training_questions_list = []

        if le is not None:
            self.le = le
        else:
            self.le = LabelEncoder()

        self.model = model

        if model:
            model._make_predict_function()

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["numpy", "tensorflow", "keras"]

    def clean_str(self, string):
        # type: (Text) -> Text
        """Tokenization/string cleaning.
        This function mostly add spaces in word contraction so they can be linked to one embedding.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py"""
        import re

        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r"\'m", " \'m", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    # computes "term frequency" which is the number of times a word appears in a document
    def tf(self, word, blob):
        return (float)(blob.words.count(word)) / (float)(len(blob.words))

    # returns the number of documents containing word
    def n_containing(self, word, bloblist):
        return sum(1 for blob in bloblist if word in blob)

    def build_tfidf_map(self):
        import json
        import numpy as np

        # manual load of training data
        with open('training_data/loreal_with_answers.json') as json_data:
            training_questions = json.load(json_data)

        questions = []
        questions_raw = []
        questions_mask = []
        answers = []
        pred_link = []
        answer_link = []
        tfidf_questions = []
        tfidf_answers = []
        for i, answer in enumerate(training_questions["faq_question"]):
            temp = []
            temp2 = []
            temp_string = ""
            answers.append([i for j in training_questions["faq_question"][answer]])
            #tfidf_answers.append(i)
            tfidf_answers.append([i for j in training_questions["faq_question"][answer]])
            pred_link.append(training_questions["faq_question"][answer][0])
            for question in training_questions["faq_question"][answer]:
                questions_mask.append(i)
                temp.append(self.clean_str(question))
                temp2.append(question)
                temp_string += " " + self.clean_str(question)
                answer_link.append(answer)
                tfidf_questions.append(tb(self.clean_str(question)))
            questions.append(temp)
            questions_raw.append(temp2)
            #tfidf_questions.append(tb(temp_string))

        answers = np.concatenate(answers)
        questions = np.concatenate(questions)
        questions_raw = np.concatenate(questions_raw)
        tfidf_answers = np.concatenate(tfidf_answers)

        return tfidf_questions

    # computes "inverse document frequency" which measures how common a word is among all documents in bloblist
    def idf(self, word, bloblist):
        return math.log(len(bloblist) / (float)(1 + self.n_containing(word, bloblist)))

    # computes the TF-IDF score. It is simply the product of tf and idf
    def tfidf(self, word, blob, bloblist):
        return self.tf(word, blob) * self.idf(word, bloblist)

    def tfidf_extract(self, questions_list, tfidf_threshold=0.10, textblob=False):
        tfidf_extract = []
        tfidf_questions = self.build_tfidf_map()

        for i, blob in enumerate(questions_list):
            print("Top words in document {}".format(i))
            if textblob:
                scores = {word: self.tfidf(word, blob, tfidf_questions) for word in blob.words}
            else:
                scores = {word: self.tfidf(word, tb(blob), tfidf_questions) for word in tb(blob).words}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            document_top = []
            for word, score in sorted_words[:5]:
                print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
                if score > tfidf_threshold and word not in document_top:
                    document_top.append(word)
            tfidf_extract.append(' '.join(document_top))
        return tfidf_extract

    def embeddings_weight_loading(self):
        import os
        import numpy as np

        self.embeddings_index = {}
        f = open(os.path.join('/app/dump', 'glove.6B.50d.txt'))
        # Building the dic of embeddings that will be used to encode each word with its embedding
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

        self.weights = np.zeros((len(self.embeddings_index) + 1, self.EMBEDDING_DIM))
        for i, word in enumerate(self.embeddings_index.items()):
            embedding_vector = word[1]
            if embedding_vector is not None:
                self.weights[i] = embedding_vector

    def labels_to_y(self, labels):
        # type: (List[Text]) -> np.ndarray
        """Transforms a list of strings into numeric label representation.

        :param labels: List of labels to convert to numeric representation"""

        return self.le.fit_transform(labels)

    def questions_to_x(self, questions):
        import numpy as np
        x_train = []
        for sentence in questions:
            temp = np.zeros(self.MAX_DOC_LENGTH, dtype=np.int)
            for i, word in enumerate(sentence.split(' ')):
                if word in self.embeddings_index:
                    temp[i] = self.embeddings_index.keys().index(word)
                else:
                    print("Missing word", word)
            x_train.append(temp)

        return x_train

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None
        """Train the intent classifier on a data set.

        :param num_threads: number of threads used during training time"""
        import numpy as np
        import keras
        from keras import backend as K
        from keras import Input, regularizers
        from keras.models import Sequential, Model, load_model
        from keras.utils import np_utils, to_categorical
        from keras.layers import merge, Embedding, LSTM
        from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
        from keras.layers.convolutional import Conv1D, MaxPooling1D
        from keras.layers.merge import concatenate
        from keras.callbacks import EarlyStopping

        labels = [e.get("intent") for e in training_data.intent_examples]

        self.embeddings_weight_loading()

        if len(set(labels)) < 2:
            logger.warn("Can not train an intent classifier. Need at least 2 different classes. " +
                        "Skipping training of intent classifier.")
        else:
            # self.embeddings_weight_loading()
            y = self.labels_to_y(labels)
            self.training_questions_list = [self.clean_str(example.text) for example in training_data.intent_examples]
            X = self.questions_to_x(
                self.tfidf_extract(self.training_questions_list,
                                   tfidf_threshold=self.TFIDF_THRESHOLD,
                                   textblob=False))
            X = np.array(X)

            embedding = Embedding(input_dim=self.weights.shape[0],
                                  output_dim=self.weights.shape[1],
                                  trainable=False,
                                  weights=[self.weights])

            inp = Input(shape=(self.MAX_DOC_LENGTH,))
            emb_seq = embedding(inp)

            print("Start training")

            d = Dense(64, activation='relu')(emb_seq)
            d = Flatten()(d)

            x = Dense(y.shape[0], activation='softmax')(d)

            self.model = Model(inputs=inp, outputs=x)

            adam = keras.optimizers.Adam(lr=0.0005,
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         epsilon=1e-08,
                                         decay=0.0001)
            self.model.compile(loss='sparse_categorical_crossentropy',
                              optimizer=adam,
                              metrics=['accuracy'])

            print(model.summary())

            tensorboard = keras.callbacks.TensorBoard(log_dir='./logs_tensorflow',
                                                     histogram_freq=1,
                                                     write_graph=True,
                                                     write_images=True)

            early_stopping = EarlyStopping(monitor='val_loss', patience=2)

            train_history = self.model.fit(X,
                                          y,
                                          shuffle=True,
                                          batch_size=1,
                                          epochs=12,
                                          #callbacks=[early_stopping],
                                          verbose=2) # 0=mute progress, 1=normal, 2=less info

            self.weights = None
            self.embeddings_index = None

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        """Returns the most likely intent and its probability for the input text."""
        import numpy as np

        self.embeddings_weight_loading()

        if not self.model:
            # component is either not trained or didn't receive enough training data
            intent = None
            intent_ranking = []
        else:
            print(message.text)
            X = [message.text]
            print(X)

            top_n = 3

            process_input = [self.clean_str(string) for string in X]
            process_input = np.array([self.questions_to_x(
                self.tfidf_extract(
                    [text],
                    tfidf_threshold=self.TFIDF_THRESHOLD,
                    textblob=False
                )
            ) for text in process_input])

            probas = self.model.predict(process_input[0], batch_size=1)

            ndtype = np.dtype([
                ("id".encode("ascii"), np.int, 1),
                ("proba".encode("ascii"), np.float64, 1)
            ])

            for i1, proba in enumerate(probas):
                proba = np.sort(
                    np.array(
                        [(idQ, proba[idQ]) for idQ in range(len(proba))],
                        ndtype
                    ),
                    order="proba".encode("ascii")
                )

                first_ones = proba[-top_n:]
                first_ones = first_ones[::-1] # order the list

                print("\nQ:\t", X[i1], "\n")
                #for i, guess in enumerate(first_ones, start=1):
                #    print "\t", guess[1]*100, "%", "\t\t", pred_link[guess[0]]
                intent = {
                    "name": first_ones[0][0],
                    "confidence": first_ones[0][1]
                }
                intent_ranking = [{
                    "name": guess[0],
                    "confidence": guess[1]
                } for guess in first_ones]

                message.set("intent", intent, add_to_output=True)
                message.set("intent_ranking", intent_ranking, add_to_output=True)

    def predict_prob(self, X):
        # type: (np.ndarray) -> np.ndarray
        """Given a bow vector of an input text, predict the intent label.
        Returns probabilities for all labels.

        :param X: bow of input text
        :return: vector of probabilities containing one entry for each label"""

        return [0]

    def predict(self, X):
        # type: (np.ndarray) -> Tuple[np.ndarray, np.ndarray]
        """Given a bow vector of an input text, predict most probable label.
        Returns only the most likely label.

        :param X: bow of input text
        :return: tuple of first, the most probable label and second, its
        probability"""

        return [0]

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> TFIDFClassifier
        import cloudpickle
        from keras.models import model_from_json

        if model_dir and model_metadata.get("tfidf_classifier"):
            meta = model_metadata.get("tfidf_classifier")

            # load json and create model
            with io.open(os.path.join(model_dir, meta["model_file"]), 'r') as f:
                loaded_model_json = f.read()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(os.path.join(model_dir, meta["weights_file"]))

            return TFIDFClassifier(model=loaded_model)

        else:
            return TFIDFClassifier()

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata
        necessary to load the model again."""

        model_file = os.path.join(model_dir, "tfidf_classifier_model.json")
        model_weight_file= os.path.join(model_dir, "tfidf_classifier_weights.h5")

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_file, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(model_weight_file)

        return {"tfidf_classifier": {"model_file": "tfidf_classifier_model.json",
                                     "weights_file": "tfidf_classifier_weights.h5",
                                     "version": 1}}
