from typing import Optional, Any


def find_element_of_list_of_dict_where_key_is_value(list_to_search: list[dict], key: Any, value: Any) -> Any:
    """ Find element of list of dict where key is value
    Args:
        list_to_search: list of dict
        key: key of dict
        value: value of key

    Returns:
        element of list of dict
    """
    for element in list_to_search:
        if element.get(key) == value:
            return element
    return None


number_emojis: list[str] = ['0️⃣', '1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣']


def emoji_to_number(emoji: str) -> Optional[int]:
    """ Convert emoji to number
    Args:
        emoji: emoji to convert

    Returns:
        number of emoji
    """
    if emoji not in number_emojis:
        return None
    return number_emojis.index(emoji)
