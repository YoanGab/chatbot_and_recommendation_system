# 3rd party imports
from dotenv import load_dotenv

# Built-in imports
import os

# local imports
from discord_bot import bot

if __name__ == '__main__':
    load_dotenv()
    bot.run(os.getenv('DISCORD_TOKEN'))
