import unittest

from services.nlp import Chatbot


def get_kwargs(**kwargs):
    kwargs.pop('user_id', None)
    return kwargs


chatbot = Chatbot(
    intent_methods={
        "greeting": get_kwargs,
        "goodbye": get_kwargs,
        "room": get_kwargs,
        "name": get_kwargs
    }
)
chatbot.train_model()


class TestChatbot(unittest.TestCase):
    def test_get_intent_from_greeting(self):
        message = "Hello, how are you?"
        intent, entities = chatbot.request(message=message)
        self.assertEqual(intent, "greeting")
        self.assertEqual(entities, {})

    def test_get_intent_from_goodbye(self):
        message = "Bye, have a good day!"
        intent, entities = chatbot.request(message=message)
        self.assertEqual(intent, "goodbye")
        self.assertEqual(entities, {})

    def test_get_intent_from_room(self):
        message = "I want to go to the room with a price cheaper than 100€"
        intent, entities = chatbot.request(message=message)
        self.assertEqual(intent, "room")
        self.assertEqual(entities, {"max_price": "100", "price": "100"})

        message = "I want to go to the room with a price above 100€"
        intent, entities = chatbot.request(message=message)
        self.assertEqual(intent, "room")
        self.assertEqual(entities, {"min_price": "100", "price": "100"})


if __name__ == '__main__':
    unittest.main()
