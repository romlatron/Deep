import numpy as np
import copy
import time

import torch
from torch import nn, optim
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib import patches, colors
import argparse

import terminal


def load_single_image(image_path):
    return Image.open(image_path, "r").convert('RGB')

class WindowWrapper():
    def __init__(self, x, y, width, height):
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

    def getCenter(self):
        return [self.x + (self.width / 2), self.y + (self.height / 2)]

    def isFace(self, score):
        self.face = True
        self.score = score

class Cluster:
    def __init__(self, squares):
        self.windows = squares
        self.xes = [win.x for win in self.windows]
        self.yes = [win.y for win in self.windows]
        self.xxes = [win.x + win.width for win in self.windows]
        self.yxes = [win.y + win.height for win in self.windows]
    def getTopLeft(self):
        return (min(self.xes), min(self.yes))

    def getTopRight(self):
        return (max(self.xxes), min(self.yes))

    def getBottomLeft(self):
        return (min(self.xes), max(self.yxes))

    def getBottomRight(self):
        return (max(self.xxes), max(self.yxes))

    def getWidth(self):
        return max(self.xxes) - min(self.xes)

    def getHeight(self):
        return max(self.yxes) - min(self.yes)

    def getRadius(self):
        return (self.getWidth() + self.getHeight()) / 4

    def getCenter(self):
        return (max(self.xxes) + min(self.xes)) / 2, (max(self.yxes) + min(self.yes)) / 2

    def getCenterWindow(self):
        minDist = float('inf')
        minWin = False
        rc = np.array(self.getCenter())
        for win in self.windows:
            c = np.array(win.getCenter())
            dist = np.linalg.norm(c-rc)
            if dist < minDist:
                minDist = dist
                minWin = win
        return minWin

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

                windows.append(WindowWrapper(pos[0], pos[1], square_len, square_len))

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

def train_net(net, loaders, n_epoch, dataset_sizes, cuda=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=.001, momentum=.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    since = time.time()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    for iteration in range(n_epoch):
        running_loss = .0
        running_hits = 0
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                net.train()
            else:
                net.eval()

            for i, data in enumerate(loaders[phase], 0):
                inputs, labels = data
                if cuda:
                    inputs, labels = inputs.to("cuda:0"), labels.to("cuda:0")

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_hits += torch.sum(preds == labels.data)

                epoch_acc = running_hits.double() / dataset_sizes[phase]

                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f acc: %.3f' % (iteration + 1, i + 1, running_loss / 2000, epoch_acc))
                    running_loss = 0.0

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(net.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net

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

def visualize_model(model, dataloaders, class_names, cuda=False, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs, labels = inputs.to("cuda:0" if cuda else "cpu"), labels.to("cuda:0" if cuda else "cpu")

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                show_image(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def grayscale_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

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
    parser.add_argument('--outliers', help='Will preview a batch of images from the test set', action='store_true')
    parser.add_argument('-e', '--test', help='A folder with labeled testing data')
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
    )

    classes = ('noface', 'face')
    net = torchvision.models.resnet18(pretrained=True)
    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features, len(classes))


    if torch.cuda.is_available():
        net.to("cuda:0")
        print("On GPU!")

    if args.train is None and args.model is None:
        print("You have to specify a model or a train set to use the net")
    elif args.train is not None and args.test is not None:
        train_set = torchvision.datasets.ImageFolder(root=args.train, transform=transform, loader=grayscale_loader)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
        test_set = torchvision.datasets.ImageFolder(root=args.test, transform=transform, loader=grayscale_loader)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True, num_workers=2)
        if args.training_preview:
            preview(train_loader, classes)

        net.load_state_dict(torch.load(args.model))
        print("Loaded model from {}".format(args.model))

        #net = train_net(net, {'train': train_loader, 'val': test_loader}, 2,
        #          {'train': len(train_set), 'val': len(test_set)}, torch.cuda.is_available())

        #if args.model is not None:
        #    torch.save(net.state_dict(), args.model)
        #    print("Saved model at {}".format(args.model))

        visualize_model(net, {'train': train_loader, 'val': test_loader}, classes, torch.cuda.is_available())
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
    window_array = image_mover(img, 0.4, 0.6, terminate_size=max(np.sqrt(img.size[0] * img.size[1] * 0.00138), 36))

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    count = 0
    lowest = float("inf")
    highest = 0
    with torch.no_grad():
        total_iters = 0
        for window_wrap in window_array:
            crop_real = transform(transforms.functional.crop(img, window_wrap.y, window_wrap.x, window_wrap.height, window_wrap.width))
            if torch.cuda.is_available():
                crop_real = crop_real.to("cuda:0")
            outputs = net(crop_real.unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            if outputs.data[0][1] > 1.2 and predicted[0] == 1:
                window_wrap.isFace(outputs.data[0][1])
                if outputs.data[0][1] > highest:
                    highest = outputs.data[0][1]

                if outputs.data[0][1] < lowest:
                    lowest = outputs.data[0][1]

                count += 1
            total_iters += 1
            terminal.print_progress(total_iters,
                                    len(window_array),
                                    prefix='Processing image: ',
                                    suffix='Complete ({} candidates)'.format(count),
                                    bar_length=80)
        print("{} faces detected".format(count))

    scan = DBSCAN(eps=min(img.size) * .08, min_samples=3)
    #for window_wrap in window_array:
        #if not window_wrap.face:
            #continue
        #rect = patches.Rectangle(window_wrap.getTopLeft(), window_wrap.width, window_wrap.height, linewidth=2,
        #                         edgecolor=colors.hsv_to_rgb((0, (lowest - window_wrap.score) / (lowest - highest), 1)), facecolor='none')
        #ax.add_patch(rect)
    matches = [i for i in window_array if i.face]
    points = np.array([i.getCenter() for i in matches])
    clusters = scan.fit(points)

    core_samples_mask = np.zeros_like(clusters.labels_, dtype=bool)
    core_samples_mask[clusters.core_sample_indices_] = True
    labels = clusters.labels_

    # # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #
    # unique_labels = set(labels)
    # color_selection = [plt.cm.Spectral(each)
    #           for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, color_selection):
    #     edge = 'k'
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]
    #         edge = 'white'
    #
    #     class_member_mask = (labels == k)
    #
    #     xy = points[class_member_mask & core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor=edge, markersize=14)
    #
    #     xy = points[class_member_mask & ~core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor=edge, markersize=6)

    class_dict = {}

    for i in range(len(matches)):
        if labels[i] not in class_dict:
            class_dict[labels[i]] = []
        class_dict[labels[i]].append(matches[i])

    if args.outliers and -1 in class_dict:
        for window_wrap in class_dict[-1]:
            rect = patches.Rectangle(window_wrap.getTopLeft(), window_wrap.width, window_wrap.height, linewidth=2,
                                     edgecolor=colors.hsv_to_rgb((0, (lowest - window_wrap.score) / (lowest - highest), 1)), facecolor='none')
            ax.add_patch(rect)

    for i in range(max(labels) + 1):
        cluster = Cluster(class_dict[i])
        rect = patches.Circle(cluster.getCenter(), cluster.getRadius(), linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
    plt.show()

if __name__ == "__main__":
    main()
