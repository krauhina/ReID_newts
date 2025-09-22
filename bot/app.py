import asyncio
import os
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from handlers.user_private import user_private_router
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

bot = Bot(token=os.getenv("TOKEN"), default=DefaultBotProperties())
bot.save_dir = 'bot/conservation'
bot.result_dir = 'bot/results'
bot.size_answer = 5

dp = Dispatcher()
dp.include_router(user_private_router)

async def on_startup():
    print("bot is started")


async def on_shutdown():
    print("bot is died")

async def main():
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())