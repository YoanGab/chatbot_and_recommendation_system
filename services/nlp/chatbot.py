# 3rd party imports
# Built-in imports
import json
import os
import pickle
import random
import re
import warnings

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Disable Tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore")


class Chatbot:
    def __init__(self, intents: dict = None, intents_path: str = "", intent_methods: dict = None,
                 model_name: str = "assistant_model", default_response: str = "I don't understand.") -> None:
        self.intents: dict = intents
        if not self.intents and intents_path.endswith(".json"):
            self.load_json_intents(intents_path)
        self.intent_methods: dict = intent_methods if intent_methods else {}
        self.model_name: str = model_name
        self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        self.default_response: str = default_response

        self.words: list[str] = []
        self.classes: list[str] = []
        self.model: Sequential = None

    def load_json_intents(self, intents_path: str) -> None:
        self.intents = json.loads(open(intents_path).read())

    def train_model(self) -> None:
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
        if model_name is None:
            model_name = self.model_name
        self.model.save(f"{model_name}.h5", self.hist)
        pickle.dump(self.words, open(f'{model_name}_words.pkl', 'wb'))
        pickle.dump(self.classes, open(f'{model_name}_classes.pkl', 'wb'))

    def load_model(self, model_name: str = None) -> None:
        if model_name is None:
            model_name = self.model_name
        self.words = pickle.load(open(f'{model_name}_words.pkl', 'rb'))
        self.classes = pickle.load(open(f'{model_name}_classes.pkl', 'rb'))
        self.model = load_model(f'{model_name}.h5')

    def _clean_up_sentence(self, sentence: str) -> list[str]:
        sentence_words: list[str] = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def _bag_of_words(self, sentence: str, words: list[str]) -> np.ndarray:
        sentence_words: list[str] = self._clean_up_sentence(sentence)
        bag: list[int] = [0] * len(words)
        for sentence_word in sentence_words:
            for index, word in enumerate(words):
                if word == sentence_word:
                    bag[index] = 1
        return np.array(bag)

    def _predict_class(self, sentence: str) -> list[dict[str: int, str: str]]:
        p: np.ndarray = self._bag_of_words(sentence, self.words)
        res: list[float] = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD: float = 0.1

        results: list[list[int, float]] = [[index, result] for index, result in enumerate(res) if
                                           result > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                'tag': self.classes[result[0]],
                'patterns': self.intents['intents'][result[0]]['patterns'],
                'probability': str(result[1])
            }
            for result in results
        ]

    def _get_response(self, intents: list[dict]) -> tuple[str, dict]:
        response: str = ''
        try:
            tag = intents[0]['tag']
            list_of_intents: list[dict] = self.intents['intents']
            for intent in list_of_intents:
                if intent['tag'] == tag:
                    response = random.choice(intent['responses'])
                    break
        except IndexError:
            tag = 'no_intent'
            response = self.default_response
        return tag, {'title': response, 'description': ''}

    def _get_variables(self, sentence: str, pattern: str) -> dict:
        # TODO: Get variables from sentence
        variables: list = re.findall(pattern, sentence)
        return {
            'variables': variables
        }

    def request(self, sentence) -> tuple[str, dict]:
        intents: list[dict] = self._predict_class(sentence=sentence)
        # return self._get_response(intents=intents)

        if intents[0]['tag'] in self.intent_methods.keys():
            variables: dict = self._get_variables(sentence=sentence, pattern=intents[0]['patterns'][0])
            return intents[0]['tag'], self.intent_methods[intents[0]['tag']]()
        else:
            return self._get_response(intents=intents)
