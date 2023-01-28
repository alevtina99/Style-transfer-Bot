from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

from config import Config, load_config

import os

from net import Net
#from func import load_image, show_image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

config: Config = load_config()
BOT_TOKEN: str = config.tg_bot.token

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)


button = KeyboardButton('Magic')
key = ReplyKeyboardMarkup(resize_keyboard=True).add(button)


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("HI! I'm a style transfer bot. Nice to meet you!\n\n"
                        "To get your photo styled, follow 2 simple steps:\n"
                        "First, send me a picture you want to style, with the caption 'Pic'\n"
                        "Then, send a picture with the desired style, with the caption 'Style'\n"
                        "Press the 'Magic' button, and let the magic happen!\n\n"
                        "Enjoy!", parse_mode=types.ParseMode.HTML)


@dp.message_handler(content_types=['photo'])
async def handle_docs_photo(message):
    await message.photo[-1].download(message.caption + str(message.from_user.id) + '.jpg')
    user_id = str(message.from_user.id)
    if os.path.exists('Pic' + user_id + '.jpg') and os.path.exists('Style' + user_id + '.jpg'):
        await bot.send_message(user_id, "Press the button and let the magic happen (it might take some time, though)", reply_markup=key)


@dp.message_handler(lambda message: message.text == "Magic")
async def process_file_command(message: types.Message):
    user_id = str(message.from_user.id)
    if os.path.exists('Pic' + user_id + '.jpg') is False:
        await bot.send_message(user_id, "Sorry, Pic is missing")
        return
    elif os.path.exists('Style' + user_id + '.jpg') is False:
        await bot.send_message(user_id, "Sorry, Style is missing")
        return

    a = Net()
    model, style_losses, content_losses, content_img = a.get_model('Style' + user_id + '.jpg'
                                                                   , 'Pic' + user_id + '.jpg')
    input_img = content_img.clone()
    output = a.run_style_transfer(model, style_losses, content_losses, input_img)

    image = output.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)

    plt.imsave('result' + user_id + '.jpg', image, format='jpg')

    await bot.send_photo(user_id, open('result' + user_id + '.jpg', 'rb'))


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
