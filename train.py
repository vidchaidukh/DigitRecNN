import random
import service
from PIL import Image

def make_test():
    l_input = []
    l_y = []
    for i in range(5005):
        l = []
        for j in range(16):
            l.append(random.choice([0, 1]))
        l_input.append(l)
        l_y.append(l.count(1))
    train_test(l_input, l_y)


def train_test(l_input, l_output):
    for i in range(5000):
        print(service.processing(l_input[i], [l_output[i]]))
    for j in range(5000, 5005):
        print("test", service.processing(l_input[j]), l_input[j], l_output[j])


print([0 for _ in range(0)] + [1] + [0 for _ in range(0, 9)])


def ttt():
    correct = 0
    for g in range(100):
        for h in range(4):
            for k in range(10):
                path_img = "ProcImg/Training30withS/" + str(k) + '_' + str(h) + '.png'
                img = Image.open(path_img)
                px = img.load()
                ListInput = [1 for i in range(901)]
                for i in range(30):
                    for j in range(30):
                        ListInput[i * 30 + j] = 0 if px[i, j] == 255 else 1

                ans = 2

                if(ans == k):
                    correct += 1
                print(k, "=", ans)
    print("correct:", correct)
