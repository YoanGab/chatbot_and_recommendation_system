# 3rd party imports
import discord
import pandas as pd
from discord.ext.commands import Bot

# Built-in imports
from random import randint
from datetime import datetime

# Local imports
from data import df
from services.nlp import Chatbot
from services.scraping import get_url

bot = Bot(command_prefix='')


def hello():
    return {
        'title': 'Bonjour',
        'description': 'Salut, comment ça va ?'
    }


def room():
    row: pd.Series = df.iloc[randint(0, df.shape[0] - 1)]
    image: str = ''
    if row['images']:
        image = row['images'].split(',')[0]
    return {
        'title': f"Getting data from room {row['id']}",
        'description': f"Title: {row['name']}\nRating: {row['rating']}",
        'url': get_url(room_id=row["id"]),
        'image': image
    }


chatbot = Chatbot(
    intents_path='intents.json',
    default_response="Sorry, I don't understand.",
    intent_methods={
        'greeting': hello,
        'room': room
    }
)
chatbot.train_model()


def get_embed_from_dict(data: dict) -> discord.Embed:
    embed: discord.Embed = discord.Embed(
        title=data['title'],
        description=data['description'] if 'description' in data else None,
        url=data['url'] if 'url' in data else ''
    )
    if 'image' in data:
        embed.set_image(url=data['image'])
    return embed


async def send_message(channel: discord.TextChannel, data: dict, with_reactions: bool = True):
    embed: discord.Embed = get_embed_from_dict(data=data)
    message_sent: discord.Message = await channel.send(embed=embed)
    if with_reactions:
        await message_sent.add_reaction('👍')
        await message_sent.add_reaction('👎')


@bot.event
async def on_ready():
    print(datetime.now())
    print('Logged in as')
    print(bot.user.name)
    print(bot.user.id)
    print('------')


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    tag, data = chatbot.request(sentence=message.content)
    with_reactions: bool = tag == 'room'
    await send_message(channel=message.channel, data=data, with_reactions=with_reactions)


@bot.event
async def on_raw_reaction_add(payload):
    if payload.user_id == bot.user.id:
        return
    if payload.emoji.name in ['👍', '👎']:
        await send_message(
            channel=bot.get_channel(payload.channel_id),
            data=room(),
            with_reactions=True
    )
