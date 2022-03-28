

def clean_room_type(room_type):
    if room_type == "home" or "appartment":
        return "Entire home/apt"
    if room_type == "hotel":
        return "Hotel room"
    if room_type == "private":
        return "Private room"
    if room_type == "shared":
        return "Shared room"


def filterBy(room_profiles, price=None, min_price=None, max_price=None, rating=None, neighbourhood_group=None,
             room_type=None, min_nights=None, rooms_to_exclude=None):
    """
    This function will filter the rooms by the user's preferences before measuring their similarity, reducing the number of rooms to compare.
    :param room_profiles: the room profiles dataframe
    :param price: a price from which we will find the room of this price ± 20€
    :param min_price: minimum price for a room
    :param max_price: maximum price for a room
    :param rating: minimum rating for a room
    :param neighbourhood_group: filters by neighbourhood_group
    :param room_type: filters by room_type
    :param min_nights: minimum number of nights for the room
    :return: the filtered room_profiles dataframe
    """
    if min_price and max_price and min_price > max_price:
        min_price, max_price = max_price, min_price
    if max_price is not None:
        room_profiles = room_profiles.loc[room_profiles.price <= max_price]
    if min_price is not None:
        room_profiles = room_profiles.loc[room_profiles.price >= min_price]
    if price is not None and min_price is None and max_price is None:
        room_profiles = room_profiles.loc[room_profiles.price <= price + 20]
        room_profiles = room_profiles.loc[room_profiles.price >= price - 20]
    if rating:
        room_profiles = room_profiles.loc[room_profiles.rating >= rating]
    if neighbourhood_group:
        neighbourhood_group += ' Island' if neighbourhood_group == 'Staten' else ""
        room_profiles = room_profiles.loc[room_profiles[neighbourhood_group] != 0.0]
    if room_type := clean_room_type(room_type):
        room_profiles = room_profiles.loc[room_profiles[room_type] != 0.0]
    if min_nights:
        room_profiles = room_profiles.loc[room_profiles.minimum_nights >= min_nights]
    if rooms_to_exclude:
        room_profiles = room_profiles.loc[room_profiles.id.isin(rooms_to_exclude) == False]
    return room_profiles
