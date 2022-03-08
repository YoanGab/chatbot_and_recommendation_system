import discord
from discord.ext.commands import Bot
from services.scraping.scrape import get_data


bot = Bot(command_prefix='')


@bot.event
async def on_ready():
    print('Logged in as')
    print(bot.user.name)
    print(bot.user.id)
    print('------')


@bot.command(
    name="room",
    help="Gives data from a room based on an ID",
)
async def room(ctx, room_id):
    data: dict = get_data(room_id=room_id)
    embed = discord.Embed(
            title=f"Getting data from room {room_id}",
            description=f"Title: {data['title']}\nRating: {data['rating']}",
            color=ctx.author.color,
            url=data["url"]
        )
    if data["images"]:
        embed.set_image(url=data["images"][0])

    await ctx.send(embed=embed)
