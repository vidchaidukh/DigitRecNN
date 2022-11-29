import telebot
from PIL import Image, ImageDraw, ImageFont
import DigitRecNN
import numpy as np

API_TOKEN = '1380825253:AAGXP6cquf9TCAK2IDsveDNV7ZoFZEdSS1A'

bot = telebot.TeleBot(API_TOKEN)

@bot.message_handler(commands=['start'])
def starter(message):
    bot.send_message(message.chat.id, "Let's go!)\nI'll send you a bg, which you'd save and then draw on it a number, "
                                      "what you wanna)\n"
                                      "And I'd try to recognize your booldraw xD")
    bot.send_document(message.chat.id, document=open('Image/canvas.jpg', 'rb'))
    bot.send_photo(message.chat.id, photo=open('Image/example.jpg', 'rb'), caption="Example)")

@bot.message_handler(commands=['help'])
def helper(message):
    bot.send_message(message.chat.id, "The reasons that make me make a wrong decision:\n\n"
                                      "1. The drawing line is too thin\n\n"
                                      "2. I have never seen handwriting like yours before\n\n"
                                      "3. You draw things that don't look like numbers at all\n\n "
                                      "If options 1 and 3 are not about you, then help me learn your handwriting: "
                                      "just write in the description of the photo what number you wrote)")


@bot.message_handler(content_types=['document', 'photo'])
def download_photos(message):
    file_obj = message.document  # if photo attached like a document
    try:
        photo_path = bot.get_file(file_obj.file_id).file_path
    except Exception:  # if photo attached just like a pic
        file_obj = message.photo[-1]  # get photo with the best resolution
        photo_path = bot.get_file(file_obj.file_id).file_path


    if message.caption is None:
        bot.send_message(message.chat.id, "Now... analyzing...")
        path_img = 'Image/test/{}_{}'.format(message.chat.id, photo_path.split("/")[1])  # make test photo's path
    else:
        bot.send_message(message.chat.id, "Kk... processing...")
        if message.caption.isdigit():
            seq_list = np.load("train_photo_counter.npy")
            digit = int(message.caption)
            photo_seq_number = seq_list[digit]
            seq_list[digit] += 1
            np.save("train_photo_counter.npy", seq_list)
            path_img = 'Image/train/{}_{}_{}.jpg'.format(digit, photo_seq_number, message.chat.id)  # make train photo's path
        else:
            bot.send_message(message.chat.id, "Oh, please, write to caption only digit from 0 to 9")
            return 0

    photo_down = bot.download_file(photo_path)         # get photo

    with open(path_img, "wb") as new_file:             # save photo
        new_file.write(photo_down)

    callAI(path_img, message.chat.id, message.caption, message.message_id)

def callAI(path_img, chat_id, digit, message_id):
    img = Image.open(path_img).convert("1")             # to black&white
    img = img.resize((30, 30), Image.LANCZOS)         # resize to 30x30
    px = img.load()
    ListInput = [1 for _ in range(900)]                 # create input list for neural network
    for i in range(30):
        for j in range(30):
            ListInput[i * 30 + j] = 0 if px[i, j] == 255 else 0
    if digit is not None:
        path_edited = "Image/train_resized/{}".format(path_img.split('/')[-1])
        img.save(path_edited, format('JPEG'))
        ans = DigitRecNN.processing(ListInput, [0 for _ in range(int(digit))] + [1] + [0 for _ in range(int(digit), 9)])
    else:
        ans = DigitRecNN.processing(ListInput)
    dig_ans = list(ans).index(max(ans))
    bot.send_message(chat_id, str(ans) + '\n' + str(dig_ans), reply_to_message_id=message_id)


bot.polling(none_stop=True, timeout=9)
