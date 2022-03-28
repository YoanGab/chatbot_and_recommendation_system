import numpy as np
import pandas as pd

from data import df
from services.recommendation.similarity import cosine_similarity
from services.recommendation.user_profile import filterBy


def normalize_df(df, cols_to_norm):
    df = df.copy()
    for col in cols_to_norm:
        df[col] = df[col] * df.rating / 5
    return df


def create_user_profile(room_profiles, user_ratings):
    """
    We create the user profile from the ratings they gave on some rooms.
    :param room_profiles: dataframe with the room profiles
    :param user_ratings: dictionnary of room_id and the rating the user gave to the room
    :return: A dataframe containing only the rooms the user liked with their rating
    """
    user_profile = pd.DataFrame()
    for room_id, rating in user_ratings.items():
        row = room_profiles.loc[room_profiles.id == room_id]
        row.loc[:, 'rating'] = rating
        user_profile = pd.concat([user_profile, row], ignore_index=True)
    return user_profile


def proceed_room_recommendation(room_profiles_norm, user_vector: pd.Series, n: int = 5) -> list:
    """
    Finds the n most recommendable rooms to the user with the given user_id.
    :param room_profiles_norm: dataframe containing the room profiles
    :param user_vector: the user vector
    :param n: the number of recommendable rooms to find
    :return: the n most recommendable rooms to the user with the given user_id
    """
    cols = df.columns[6:]
    similarity_scores = []
    for index, row in room_profiles_norm.iterrows():
        room_id = row.id
        room_vector = list(room_profiles_norm.loc[room_profiles_norm.id == room_id][cols].values[0])
        similarity_scores.append((room_id, cosine_similarity(user_vector, room_vector)))
    similarity_scores = [x for x in similarity_scores if not np.isnan(x[1])]
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    return similarity_scores[:n]


def recommend_room(user: dict) -> tuple[int, bool]:
    """
    Finds the n most recommendable rooms to the user with the given user_id.
    :param user: the user profile
    :return: the n most recommendable rooms to the user with the given user_id
    """
    print(f"Criteria: {user}")
    df_copy = df.copy()
    user_profile = create_user_profile(df_copy, user.get('ratings', {}))
    respects_criteria: bool = True
    filtered_rooms = filterBy(
        df_copy,
        min_price=user.get("min_price"),
        max_price=user.get("max_price"),
        price=user.get("price"),
        neighbourhood_group=user.get("neighbourhood"),
        room_type=user.get("room_type"),
        min_nights=user.get("minimum_nights"),
        rating=user.get("rating"),
        rooms_to_exclude=user.get('ratings', {}).keys()
    )
    if len(filtered_rooms) == 0:
        respects_criteria = False
        filtered_rooms = df_copy

    if len(user_profile) > 0:
        user_profile = normalize_df(user_profile, filtered_rooms.columns[6:])
        user_vector = user_profile[filtered_rooms.columns[6:]].mean()
        recommended_rooms = proceed_room_recommendation(filtered_rooms, user_vector, n=1)
        return recommended_rooms[0][0], respects_criteria

    return np.random.choice(filtered_rooms.id), respects_criteria
