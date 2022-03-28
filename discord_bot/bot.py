# 3rd party imports
import asyncio
# Built-in imports
from datetime import datetime
from typing import Optional

import discord
import pandas as pd
from discord.ext.commands import Bot

# Local imports
from data import df
from services.nlp import Chatbot
from services.recommendation.get_recommendation import recommend_room
from services.scraping import get_url, get_id_from_url
from services.utils import emoji_to_number, number_emojis

bot: Bot = Bot(command_prefix='')

users_profiles: dict = {}
messages_sent: dict = {}


def get_messages_for_help() -> str:
    """ Get messages for help
    :return: str
    """
    return "\n".join([
        'Give me a room cheaper than 150€',
        'My name is John',
        'Hello'
    ])


def default_response(**kwargs) -> dict:
    """ Default response
    :param kwargs: dict
    :return: dict
    """
    return {
        'title': 'Here are some messages than you can ask me',
        'description': get_messages_for_help()
    }


def save_user_profile(user_id: int, data: dict) -> None:
    """ Save user profile
    :param user_id: int
    :param data: dict
    :return: None
    """
    if user_id not in users_profiles:
        users_profiles[user_id] = data
    for key, value in data.items():
        if value is not None:
            users_profiles[user_id][key] = value
    if data.get('max_price') and data['min_price'] is None:
        users_profiles[user_id]['min_price'] = None
    if data.get('min_price') and data['max_price'] is None:
        users_profiles[user_id]['max_price'] = None
    if 'price' not in data and users_profiles[user_id].get('price'):
        users_profiles[user_id]['price'] = None


def hello(user_id: int) -> dict:
    """ Say Hello
    :param user_id: int
    :return: dict
    """
    name: str = ''
    if user_id in users_profiles and 'name' in users_profiles[user_id]:
        name = users_profiles[user_id]['name']

    return {
        'title': f'Hello {name}!',
        'description': 'How are you?'
    }


def get_name(user_id: int, name: str = '') -> dict:
    """ Store user name
    :param user_id: int
    :param name: str
    :return: dict
    """
    name = name.title()
    if name:
        save_user_profile(user_id=user_id, data={'name': name})

    return {
        'title': f'Nice to meet you {name}!',
        'description': 'I am a bot, I will help you find a room for your stay.'
    }


def room(user_id: int, min_price: int = None, max_price: int = None, price: int = None, neighbourhood: str = None,
         room_type: str = None, minimum_nights: int = None, rating: int = None) -> dict:
    """ Get room
    :param user_id: int
    :param min_price: int
    :param max_price: int
    :param price: int
    :param neighbourhood: str
    :param room_type: str
    :param minimum_nights: int
    :param rating: int
    :return: dict
    """
    save_user_profile(
        user_id=user_id,
        data={
            'min_price': int(min_price) if min_price else None,
            'max_price': int(max_price) if max_price else None,
            'price': int(price) if price else None,
            'neighbourhood': neighbourhood.title() if neighbourhood else None,
            'room_type': room_type.title() if room_type else None,
            'minimum_nights': int(minimum_nights) if minimum_nights else None,
            'rating': float(rating) if rating else None
        }
    )

    room_id, respects_criteria = recommend_room(users_profiles[user_id])
    room: pd.Series = df[df['id'] == room_id].iloc[0]

    image: str = ''
    if room['images']:
        image = room['images'].split(',')[0]
    description = f"Price: {room['price']}€\nRating: {room['rating']}" \
                  f"{'**We did not find any room that meets all your criteria.**' if not respects_criteria else ''}" \
                  f"\n\nYour criteria:\n" \
                  f"Neighbourhood: {users_profiles[user_id]['neighbourhood'] if users_profiles[user_id]['neighbourhood'] else ''}\n" \
                  f"Room type: {users_profiles[user_id]['room_type'] if users_profiles[user_id]['room_type'] else ''}\n" \
                  f"Minimum nights: {str(users_profiles[user_id]['minimum_nights']) if users_profiles[user_id]['minimum_nights'] else ''}\n" \
                  f"Minimum price: {str(users_profiles[user_id]['min_price']) if users_profiles[user_id]['min_price'] else ''}\n" \
                  f"Maximum price: {str(users_profiles[user_id]['max_price']) if users_profiles[user_id]['max_price'] else ''}\n" \
                  f"Mean Price: {str(users_profiles[user_id]['price']) if users_profiles[user_id]['price'] and not users_profiles[user_id]['max_price'] and not users_profiles[user_id]['max_price'] else ''}\n" \
                  f"Min Rating: {str(users_profiles[user_id]['rating']) if users_profiles[user_id]['rating'] else ''}"

    return {
        'title': f"Room {room['id']}: {room['name']}",
        'description': description,
        'url': get_url(room_id=room['id']),
        'image': image,
        'fields': {'id': room['id']}
    }


def saved_rooms(user_id: int) -> dict:
    """ Get saved rooms
    :param user_id: int
    :return: dict
    """
    fields: dict = {}
    if user_id in users_profiles and 'ratings' in users_profiles[user_id]:
        sorted_ratings = {k: v for k, v in sorted(users_profiles[user_id]['ratings'].items(), key=lambda item: item[1], reverse=True)}
        for room_id, rating in sorted_ratings.items():
            room: pd.Series = df[df['id'] == room_id].iloc[0]
            fields[f"{room['name']}"] = f"[Rating of {rating}]({get_url(room_id=room_id)})"

    return {
        'title': 'Here are the rooms you saved',
        'description': 'You can ask me to recommend a room for you',
        'fields': fields
    }


chatbot: Chatbot = Chatbot(
    default_response=default_response(),
    intent_methods={
        'greeting': hello,
        'room': room,
        'name': get_name,
        'help': default_response,
        'saved_rooms': saved_rooms
    }
)
chatbot.train_model()


def get_embed_from_dict(data: dict) -> discord.Embed:
    """ Get embed from dict
    :param data: dict
    :return: discord.Embed
    """
    embed: discord.Embed = discord.Embed(
        title=data['title'],
        description=data['description'] if 'description' in data else None,
        url=data['url'] if 'url' in data else ''
    )
    if 'image' in data:
        embed.set_image(url=data['image'])
    if 'fields' in data:
        for key, value in data['fields'].items():
            embed.add_field(name=key, value=value)
    return embed


async def react_with_emojis(message: discord.Message) -> None:
    """ React with emojis
    :param message: discord.Message
    :return: None
    """
    await asyncio.gather(*[
        message.add_reaction(emoji=emoji)
        for emoji in number_emojis
    ])


async def send_message(channel: discord.TextChannel, message_dict: dict, with_reactions: bool = True) -> None:
    """ Send message
    :param channel: discord.TextChannel
    :param message_dict: dict
    :param with_reactions: bool
    :return: None
    """
    embed: discord.Embed = get_embed_from_dict(data=message_dict)
    message_sent: discord.Message = await channel.send(embed=embed)
    if with_reactions:
        await react_with_emojis(message=message_sent)


async def reply(message: discord.Message, message_dict: dict, with_reactions: bool = True) -> None:
    """ Reply
    :param message: discord.Message
    :param message_dict: dict
    :param with_reactions: bool
    :return: None
    """
    message_sent: discord.Message = await message.reply(embed=get_embed_from_dict(data=message_dict))
    if with_reactions:
        await react_with_emojis(message=message_sent)


async def get_room_from_message(message_id: int, channel_id: int) -> Optional[int]:
    """ Get room from message
    :param message_id: int
    :param channel_id: int
    :return: int
    """
    channel: discord.TextChannel = bot.get_channel(channel_id)
    message: discord.Message = await channel.fetch_message(message_id)
    embed: discord.Embed = message.embeds[0]
    return get_id_from_url(embed.url)


async def save_rating(user_id: int, message_id: int, channel_id: int, rating_emoji: str) -> None:
    """ Save rating
    :param channel_id:
    :param user_id: int
    :param message_id: int
    :param rating_emoji: str
    :return: None
    """
    rating: int = emoji_to_number(emoji=rating_emoji)
    room_id: int = await get_room_from_message(message_id=message_id, channel_id=channel_id)
    if room_id:
        if user_id not in users_profiles:
            users_profiles[user_id] = {}
        if 'ratings' not in users_profiles[user_id]:
            users_profiles[user_id]['ratings'] = {}
        users_profiles[user_id]['ratings'][room_id] = rating


@bot.event
async def on_ready() -> None:
    """ On ready
    :return: None
    """
    print(datetime.now())
    print('Logged in as')
    print(bot.user.name)
    print(bot.user.id)
    print('------')


@bot.event
async def on_message(message: discord.Message) -> None:
    """ On message
    :param message: discord.Message
    :return: None
    """
    if message.author == bot.user:
        return
    intent, message_dict = chatbot.request(user_id=message.author.id, message=message.content)
    with_reactions: bool = intent == 'room'
    await reply(message=message, message_dict=message_dict, with_reactions=with_reactions)


@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
    """ On raw reaction add
    :param payload: discord.RawReactionActionEvent
    :return: None
    """
    if payload.user_id == bot.user.id:
        return
    if payload.emoji.name in number_emojis:
        await save_rating(user_id=payload.user_id, message_id=payload.message_id,
                          channel_id=payload.channel_id, rating_emoji=payload.emoji.name)
        await send_message(
            channel=bot.get_channel(payload.channel_id),
            message_dict=room(user_id=payload.user_id),
            with_reactions=True
        )
