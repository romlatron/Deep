import sys
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import glob

def processType(type):
    image_list = []
    if not os.path.exists('{}/out'.format(sys.argv[1])):
        os.mkdir('{}/out'.format(sys.argv[1]))
    for filename in glob.glob('{}/*.{}'.format(sys.argv[1], type)):
        try:
            im=Image.open(filename).convert('LA').convert('RGB')
            square = min(im.size)
            offset = int((max(im.size) - square) / 2)
            # Wide
            if im.size[0] > im.size[1]:
                im = im.crop((offset, 0, im.size[0] - offset, square))
            #High
            else:
                im = im.crop((0, offset, square, im.size[1] - offset))
            im = im.resize((36,36))
            image_list.append(im)
            # plt.imshow(image_list[-1])
            # plt.show()
            im.save('{}/out/{}.jpg'.format(sys.argv[1], time.time()))
        except OSError:
            print("Ba dum ts")


if __name__ == "__main__":
    types = ["jpeg", "jpg", "gif", "png"]
    for type in types:
        processType(type)