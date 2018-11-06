import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import patches, colors
import argparse


def load_single_image(image_path):
    return Image.open(image_path, "r").convert('L')

class WindowWrapper():
    def __init__(self, image, x, y, width, height):
        self.data = image
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.face = False
        self.score = 0

    def getTopLeft(self):
        return (self.x, self.y)

    def getTopRight(self):
        return (self.x + self.width, self.y)

    def getBottomLeft(self):
        return (self.x, self.y + self.height)

    def getBottomRight(self):
        return (self.x + self.width, self.y + self.height)

    def isFace(self, score):
        self.face = True
        self.score = score


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc0 = nn.Linear(32, 576)
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def show_image(img):
    img = img / 2 + .5
    numpy_img = img.numpy()
    plt.imshow(np.transpose(numpy_img, (1, 2, 0)))
    plt.show()

def preview(loader, classes):
    data_iterator = iter(loader)
    images, labels = data_iterator.next()
    show_image(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

def image_mover(pil_image, move_rate, shrink_factor, terminate_size=36, debug=False):
    # Define window size here
    square_len = min(pil_image.size)
    windows = []

    # While window size is not below our terminate size (training images), continue to shrink it
    while square_len >= terminate_size:
        # Reset position for new scanning iteration
        pos = [0, 0]

        # While vertical square boundaries do not exceed image, continue to image
        while square_len + pos[1] <= pil_image.size[1]:
            # While vertical square boundaries do not exceed image, continue to row
            while square_len + pos[0] <= pil_image.size[0]:
                if debug:
                    print("Snipping x {} y {} at square size {} \t (scale factor {})".format(pos[0], pos[1], square_len, 64.0 / square_len))

                crop = transforms.functional.resized_crop(pil_image, pos[1], pos[0], square_len, square_len, terminate_size)
                windows.append(WindowWrapper(crop, pos[0], pos[1], square_len, square_len))

                if debug:
                    plt.imshow(windows[-1].data)
                    plt.show()

                # Move square to the right (move rate is in the interval ]0;1])
                pos[0] += max(int(square_len * move_rate), 2)

            # Scan of the row has been completed, reset row position
            pos[0] = 0
            # Move to an overlapping row below (move rate is in the interval ]0;1])
            pos[1] += max(int(square_len * move_rate), 2)

        # Shrink square (shrink rate ]0;1[) after image scan with current size terminated
        square_len *= shrink_factor
    return windows

def train_net(net, train_loader, n_epoch, cuda=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=.001, momentum=.9)
    for iteration in range(n_epoch):
        last_loss = float("inf")
        running_loss = .0
        early_stop = False
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            if cuda:
                inputs, labels = inputs.to("cuda:0"), labels.to("cuda:0")

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (iteration + 1, i + 1, running_loss / 2000))
                if last_loss < running_loss:
                    # early_stop = True
                    # break
                    pass
                last_loss = running_loss
                running_loss = 0.0
        if early_stop:
            print("Stopping early")
            break
    print('Finished Training')

def test_net(net, test_loader, classes, cuda=False):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if cuda:
                images, labels = images.to("cuda:0"), labels.to("cuda:0")

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if cuda:
                images, labels = images.to("cuda:0"), labels.to("cuda:0")

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        recall = 100*class_correct[i] / class_total[i]
        precision = 100 * class_correct[i] / total
        print('Precision of %5s : %2d %%' % (
            classes[i], precision))
        print('Recall of %5s : %2d %%' % (
            classes[i], recall))
        print('F1 score of %5s : %2d %%' % (
            classes[i], 2 * precision*recall/(precision+recall)))

def grayscale_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def main():
    parser = argparse.ArgumentParser(description='A deep learning powered face recognition app.',
                                     epilog='Written for the INSA Lyon DEEP Project by Anh Pham, Mathilde du Plessix, '
                                            'Romain Latron, Beonît Zhong, Martin Haug. 2018 All rights reserved.')

    parser.add_argument('image', help='Image to recognize faces on')
    parser.add_argument('--epoch', help='Number of epoch', type=int, default=5)
    parser.add_argument('-m', '--model', help='Pass a model file to skip training')
    parser.add_argument('-t', '--train', help='A folder with correctly labeled training data. '
                                              'Will save at model path if option is specified.')
    parser.add_argument('--training-preview', help='Will preview a batch of images from the training set', action='store_true')
    parser.add_argument('--test-preview', help='Will preview a batch of images from the test set', action='store_true')
    parser.add_argument('-e', '--test', help='A folder with labeled testing data')
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
    )

    classes = ('noface', 'face')
    net = Net()
    if torch.cuda.is_available():
        net.to("cuda:0")
        print("On GPU!")

    if args.train is None and args.model is None:
        print("You have to specify a model or a train set to use the net")
    elif args.train is not None:
        train_set = torchvision.datasets.ImageFolder(root=args.train, transform=transform, loader=grayscale_loader)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
        if args.training_preview:
            preview(train_loader, classes)
        train_net(net, train_loader, args.epoch, torch.cuda.is_available())
        if args.model is not None:
            torch.save(net.state_dict(), args.model)
            print("Saved model at {}".format(args.model))
    else:
        net.load_state_dict(torch.load(args.model))
        print("Loaded model from {}".format(args.model))

    if args.test is not None:
        test_set = torchvision.datasets.ImageFolder(root=args.test, transform=transform, loader=grayscale_loader)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True, num_workers=2)
        if args.test_preview:
            preview(test_loader, classes)

        test_net(net, test_loader, classes, torch.cuda.is_available())

    img = load_single_image(args.image)
    window_array = image_mover(img, 0.4, 0.6)

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    count = 0
    lowest = float("inf")
    highest = 0
    for window_wrap in window_array:
        crop_real = transform(window_wrap.data)
        if torch.cuda.is_available():
            crop_real = crop_real.to("cuda:0")
        outputs = net(crop_real.unsqueeze(0))
        _, predicted = torch.max(outputs.data, 1)
        if outputs.data[0][1] > 2.6 and predicted[0] == 1:
            window_wrap.isFace(outputs.data[0][1])
            if outputs.data[0][1] > highest:
                highest = outputs.data[0][1]

            if outputs.data[0][1] < lowest:
                lowest = outputs.data[0][1]

            print("Face detected")
            count += 1
    print("{} faces detected".format(count))

    rects = {}
    for window_wrap in window_array:
        if not window_wrap.face:
            continue
        rect = patches.Rectangle(window_wrap.getTopLeft(), window_wrap.width, window_wrap.height, linewidth=2,
                                 edgecolor=colors.hsv_to_rgb((0, (lowest - window_wrap.score) / (lowest - highest), 1)), facecolor='none')
        rects[rect] = window_wrap.score
        ax.add_patch(rect)

    keys = list(rects.keys())
    for i in range(len(keys)):
        rect = keys[i]
        for j in range(i+1,len(keys)):
            rect2 = keys[j]
            path = rect.get_patch_transform().transform_path(rect.get_path())
            path2 = rect2.get_patch_transform().transform_path(rect2.get_path())
            pc = path2.intersects_path(path) | path2.contains_path(path) | path.contains_path(path2)
            if pc and rects[rect]>rects[rect2]:
                rect.set_linewidth(2)
                rect2.set_linewidth(0.5)
            elif pc and rects[rect]<rects[rect2]:
                rect.set_linewidth(0.5)
                rect2.set_linewidth(2)
            elif pc and rects[rect]==rects[rect2]:
                rect.set_linewidth(1)
                rect2.set_linewidth(1)


    plt.show()

if __name__ == "__main__":
    main()
