from aiogram import Router, types, F, Bot
from aiogram.filters import CommandStart
from bot.handlers.model_utils import photo_processing
import os

user_private_router = Router()


@user_private_router.message(CommandStart())
async def start(message: types.Message):
    await message.answer(
        'Данный бот создан для демонстрации возможностей модели.\nПожалуйста, загрузите фотографию брюшка тритона, и модель выведет вам 5 самых похожих изображений')


@user_private_router.message(F.photo)
async def handle_photo(message: types.Message, bot: Bot):
    try:
        photo = message.photo[-1]
        photo_file = await bot.get_file(photo.file_id)
        filename = "image.jpg"
        filepath = os.path.join(bot.save_dir, filename)

        await bot.download_file(photo_file.file_path, filepath)

        await message.answer(
            '✅ Ваша фотография была успешно сохранена!\n⌛ Пожалуйста, подождите, пока завершится работа алгоритма')

        success = await photo_processing(bot)

        if success:
            result_files = [os.path.join(bot.result_dir, f"top{i}.jpg") for i in range(1, bot.size_answer+1)]
            existing_files = [f for f in result_files if os.path.exists(f)]

            if not existing_files:
                await message.answer("❌ Результаты обработки не найдены")
                return

            res_file_path = os.path.join(bot.result_dir, "res.txt")
            if os.path.exists(res_file_path):
                with open(res_file_path, 'r', encoding='utf-8') as f:
                    res_str = f.read()

                media_group = []
                for i, filepath in enumerate(existing_files[:bot.size_answer]):
                    with open(filepath, 'rb') as f:
                        caption = f"Результаты обработки:\n\n{res_str}" if i == 0 else None
                        media_group.append(types.InputMediaPhoto(
                            media=types.BufferedInputFile(f.read(), filename=f"top{i + 1}.jpg"),
                            caption=caption
                        ))

                await message.answer_media_group(media=media_group)
                await message.answer("✅ Готово! Вот 5 самых похожих изображений")
            else:
                await message.answer("❌ Файл с результатами не найден")
        else:
            await message.answer("❌ Ошибка при обработке изображения")

    except Exception as e:
        await message.answer(f"❌ Ошибка: {str(e)}")