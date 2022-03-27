import numpy as np


def cosine_similarity(user_vector: list, room_vector: list) -> float:
    """
    Calculates the cosine similarity between two users.
    :param user_vector: user vector to compare to the room vector
    :param room_vector: room vector
    :return: the cosine similarity
    """
    if len(user_vector) != len(room_vector):
        raise ValueError('Vectors must have the same length')
    return np.abs(np.dot(user_vector, room_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(room_vector)))
