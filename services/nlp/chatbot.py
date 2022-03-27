import json
import os
import pickle
import random
import re
import warnings
from typing import Optional, Union

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

from services.utils.utils import find_element_of_list_of_dict_where_key_is_value

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Disable Tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore")


class Chatbot:
    def __init__(self, intent_methods: dict = None, model_name: str = "assistant_model",
                 default_response=None, min_probability: float = 0.8) -> None:
        """ Initialize the Chatbot object.
        :param intents: The intents dictionary.
        :param intents_path: The path to the intents file.
        :param intent_methods: The methods to be used for the intents.
        :param model_name: The name of the model.
        :param default_response: The default response.
        :param min_probability: The minimum probability for the intents.
        :return: None
        """
        if default_response is None:
            default_response = {"title": "I don't understand."}

        self.intents: dict = self.load_json_intents()
        self.intent_methods: dict = intent_methods if intent_methods else {}
        self.model_name: str = model_name
        self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        self.default_response: dict = default_response

        self.words: list[str] = []
        self.classes: list[str] = []
        self.model: Sequential = None

        self.MIN_PROBABILITY: float = min_probability

    @staticmethod
    def load_json_intents() -> dict:
        """ Load the intents from a json file.
        :return: The intents.
        """
        with open('./services/nlp/intents.json') as json_data:
            intents: dict = json.load(json_data)
        return intents

    def train_model(self) -> None:
        """ Train the model.
        :return: None
        """
        self.words = []
        self.classes = []
        documents: list[tuple] = []
        ignore_letters: list[str] = ['!', '?', ',', '.']

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                words: list[str] = nltk.word_tokenize(pattern)
                self.words.extend(words)
                documents.append((words, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(word.lower()) for word in self.words if word not in ignore_letters]
        self.words = sorted(list(set(self.words)))

        self.classes = sorted(list(set(self.classes)))

        training: list[list] = []
        output_empty: list[int] = [0] * len(self.classes)

        for document in documents:
            bag: list = []
            tuple_word_patterns: tuple = document[0]
            word_patterns: list[str] = [self.lemmatizer.lemmatize(word.lower()) for word in tuple_word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row: list[int] = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training: np.ndarray = np.array(training, dtype=object)

        train_x: list[list[int]] = list(training[:, 0])
        train_y: list[list[int]] = list(training[:, 1])

        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(train_y[0]), activation='softmax'))

        sgd: SGD = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.hist = self.model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=0)

    def save_model(self, model_name: str = None) -> None:
        """ Save the model.
        :param model_name: The name of the model.
        :return: None
        """
        if model_name is None:
            model_name = self.model_name
        self.model.save(f"{model_name}.h5", self.hist)
        pickle.dump(self.words, open(f'{model_name}_words.pkl', 'wb'))
        pickle.dump(self.classes, open(f'{model_name}_classes.pkl', 'wb'))

    def load_model(self, model_name: str = None) -> None:
        """ Load the model.
        :param model_name: The name of the model.
        :return: None
        """
        if model_name is None:
            model_name = self.model_name
        self.words = pickle.load(open(f'{model_name}_words.pkl', 'rb'))
        self.classes = pickle.load(open(f'{model_name}_classes.pkl', 'rb'))
        self.model = load_model(f'{model_name}.h5')

    def _clean_up_sentence(self, message: str) -> list[str]:
        """ Clean up the message.
        :param message: The message to clean up.
        :return: The cleaned up message.
        """
        sentence_words: list[str] = nltk.word_tokenize(message)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def _bag_of_words(self, message: str, words: list[str]) -> np.ndarray:
        """ Create the bag of words.
        :param message: The message to create the bag of words for.
        :param words: The words to create the bag of words for.
        :return: The bag of words.
        """
        sentence_words: list[str] = self._clean_up_sentence(message)
        bag: list[int] = [0] * len(words)
        for sentence_word in sentence_words:
            for index, word in enumerate(words):
                if word == sentence_word:
                    bag[index] = 1
        return np.array(bag)

    def _predict_class(self, message: str) -> list[dict[str: int, str: str]]:
        """ Predict the class of the message.
        :param message: The message to predict the class of.
        :return: The predicted class.
        """
        p: np.ndarray = self._bag_of_words(message, self.words)
        res: list[float] = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD: float = self.MIN_PROBABILITY

        results: list[list[int, float]] = [[index, result] for index, result in enumerate(res) if
                                           result >= ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                'tag': self.classes[result[0]],
                'entities': find_element_of_list_of_dict_where_key_is_value(
                    list_to_search=self.intents['intents'],
                    key='tag',
                    value=self.classes[result[0]]
                )['entities'],
                'probability': str(result[1])
            }
            for result in results
        ]

    def _get_response(self, intents: list[dict]) -> tuple[str, dict]:
        """ Get the response.
        :param intents: The intents to get the response for.
        :return: The response.
        """
        response: Optional[Union[dict, str]] = None
        try:
            tag: str = intents[0]['tag']
            list_of_intents: list[dict] = self.intents['intents']
            for intent in list_of_intents:
                if intent['tag'] == tag:
                    response = {
                        'title': random.choice(intent['responses']),
                        'description': ''
                    }
                    break
        except IndexError:
            tag = 'no_intent'
            response = self.default_response
        return tag, response

    @staticmethod
    def _get_entities(message: str, intent: dict) -> dict:
        """ Get the entities.
        :param message: The message to get the entities for.
        :param intent: The intent to get the entities for.
        :return: The entities.
        """
        entities_to_find: dict[str: str] = intent['entities']
        entities: dict = {}
        for entity, regexes in entities_to_find.items():
            for regex in regexes:
                regex = r"{}".format(regex)
                regex_match: re.Match = re.search(regex, message, re.IGNORECASE)
                if regex_match:
                    entities[entity] = regex_match.group(entity)
                    break
        return entities

    def request(self, message: str, user_id: Optional[int] = None) -> tuple[str, dict]:
        """ Request the response.
        :param user_id: The user id.
        :param message: The message to request the response for.
        :return: The response.
        """
        intents: list[dict] = self._predict_class(message=message)

        if intents and intents[0]['tag'] in self.intent_methods.keys():
            entities: dict = self._get_entities(message=message, intent=intents[0])
            return intents[0]['tag'], self.intent_methods[intents[0]['tag']](user_id=user_id, **entities)
        else:
            return self._get_response(intents=intents)
