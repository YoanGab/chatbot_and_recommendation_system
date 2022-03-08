import requests
from bs4 import BeautifulSoup


def get_url(room_id: int) -> str:
    """ Gets the url of the room
    :param room_id: int, the room id
    """
    return f"https://www.airbnb.com/rooms/{room_id}"


def get_html(url: str) -> BeautifulSoup:
    """ Gets the html of the page
    :param url: str, the url of the page
    """
    page = requests.get(url)
    return BeautifulSoup(page.content, 'html.parser')


def get_images(soup: BeautifulSoup) -> list:
    """ Gets the picture url from a given url
    :param soup: BeautifulSoup object, the html of the page
    """
    return [img.get('src') for img in soup.find_all('img')]


def get_title(soup: BeautifulSoup) -> str:
    """ Gets the title of the room
    :param soup: BeautifulSoup object, the html of the page
    """
    return soup.find('h1').text


def get_mean_ratings(soup: BeautifulSoup) -> float:
    """ Gets the mean ratings of the room
    :param soup: BeautifulSoup object, the html of the page
    """
    return float(soup.find("span", {"class": "_12si43g"}).text.replace("·", ""))


def get_data(room_id: int, response: str = None) -> dict:
    """ Gets the data from the url
    :param room_id: int, the room id
    :param response: str, the html of the page
    """

    url: str = get_url(room_id)
    if response is None:
        soup = get_html(url=url)
    else:
        soup = BeautifulSoup(response, 'html.parser')

    try:
        return {
            'id': room_id,
            'url': url,
            'title': get_title(soup),
            'images': get_images(soup),
            'rating': get_mean_ratings(soup)
        }
    except Exception as e:
        return {
            'id': room_id,
            'url': url,
            'title': "",
            'images': [],
            'rating': None
        }
