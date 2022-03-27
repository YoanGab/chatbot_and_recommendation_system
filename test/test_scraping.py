import unittest

from services.scraping.scrape import get_url, get_data


class TestScraping(unittest.TestCase):
    def test_get_url(self):
        self.assertEqual(
            get_url(room_id=25),
            "https://www.airbnb.com/rooms/25"
        )

    def test_get_url_with_invalid_room_id(self):
        self.assertEqual(
            get_url(room_id=-1),
            ""
        )

    def test_get_url_with_invalid_room_id_type(self):
        self.assertEqual(
            get_url(room_id=None),
            ""
        )

    def test_get_data(self):
        self.assertEqual(
            get_data(room_id=2595),
            {
                'id': 2595,
                'url': 'https://www.airbnb.com/rooms/2595',
                'title': 'Skylit Midtown Castle',
                'images': [
                    'https://a0.muscache.com/im/pictures/f0813a11-40b2-489e-8217-89a2e1637830.jpg?im_w=720',
                    'https://a0.muscache.com/im/pictures/e5299666-37a9-4e39-b2c8-54ca930ae3a8.jpg?im_w=720',
                    'https://a0.muscache.com/im/pictures/98f1c212-80f2-4f67-9020-ce88369e233e.jpg?im_w=720',
                    'https://a0.muscache.com/im/pictures/44e6dc68-ad3a-4862-8d82-b9d7d1dcabda.jpg?im_w=720',
                    'https://a0.muscache.com/im/pictures/e22224e8-a6f7-4904-a23a-8bfcf579dc68.jpg?im_w=720',
                    'https://a0.muscache.com/im/pictures/44e6dc68-ad3a-4862-8d82-b9d7d1dcabda.jpg?im_w=720'
                ],
                'rating': 4.7
            }
        )

    def test_get_data_with_invalid_room_id(self):
        self.assertEqual(
            get_data(room_id=25),
            {
                'id': 25,
                'url': "https://www.airbnb.com/rooms/25",
                'title': '',
                'images': [],
                'rating': None
            }
        )


if __name__ == '__main__':
    unittest.main()
