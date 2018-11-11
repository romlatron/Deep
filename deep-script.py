import copy
import time
import itertools
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, colors
from sklearn.cluster import DBSCAN

import torch
from torch import nn, optim

import torchvision
from torchvision import transforms
from PIL import Image

import terminal


# Loads image in color and B/W
def load_single_image(image_path):
    return Image.open(image_path, "r"), Image.open(image_path, "r").convert('L').convert('RGB')


# Represents metadata of an image crop. Can calculate several coordinates relevant to crop
class WindowWrapper:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.face = False
        self.score = 0

    def get_top_left(self):
        return self.x, self.y

    def get_top_right(self):
        return self.x + self.width, self.y

    def get_bottom_left(self):
        return self.x, self.y + self.height

    def get_bottom_right(self):
        return self.x + self.width, self.y + self.height

    def get_center(self):
        return [self.x + (self.width / 2), self.y + (self.height / 2)]

    def is_face(self, score):
        self.face = True
        self.score = score


# Contains WindowWrappers belonging to a cluster. Can calculate several coordinates relevant to that cluster
class Cluster:
    def __init__(self, squares):
        self.windows = squares
        self.left_x_list = [win.x for win in self.windows]
        self.top_y_list = [win.y for win in self.windows]
        self.right_x_list = [win.x + win.width for win in self.windows]
        self.bottom_y_list = [win.y + win.height for win in self.windows]

    def get_top_left(self):
        return min(self.left_x_list), min(self.top_y_list)

    def get_top_right(self):
        return max(self.right_x_list), min(self.top_y_list)

    def get_bottom_left(self):
        return min(self.left_x_list), max(self.bottom_y_list)

    def get_bottom_right(self):
        return max(self.right_x_list), max(self.bottom_y_list)

    def get_width(self):
        return max(self.right_x_list) - min(self.left_x_list)

    def get_height(self):
        return max(self.bottom_y_list) - min(self.top_y_list)

    def get_radius(self):
        return (self.get_width() + self.get_height()) / 4

    def get_center(self):
        return (max(self.right_x_list) + min(self.left_x_list)) / 2,\
               (max(self.bottom_y_list) + min(self.top_y_list)) / 2

    def get_center_window(self):
        min_dist = float('inf')
        min_win = False
        rc = np.array(self.get_center())
        for win in self.windows:
            c = np.array(win.get_center())
            dist = np.linalg.norm(c-rc)
            if dist < min_dist:
                min_dist = dist
                min_win = win
        return min_win

    def get_max_score(self):
        return max([x.score for x in self.windows])

    def get_min_score(self):
        return min([x.score for x in self.windows])

    def get_avg_score(self):
        return sum([x.score for x in self.windows]) / len(self.windows)


# for one node in a dictionary representing a graph, find all connected nodes
def find_group_recursive(thing, dictionary):
    if thing not in dictionary:
        return [thing]
    else:
        res = []
        for overlapping in dictionary[thing]:
            res.extend(find_group_recursive(overlapping, dictionary))
        res.append(thing)
        return res


# Given a dictionary which indicates directional edges in a graph, return all connected subgraphs seperately
def get_groups(dictionary):
    ignore = []
    res = []
    for key, _ in dictionary.items():
        if key in ignore:
            continue
        group = find_group_recursive(key, dictionary)
        ignore.extend(group)
        res.append(group)
    return res


# Plot an image
def show_image(img):
    img = img / 2 + .5
    numpy_img = img.numpy()
    plt.imshow(np.transpose(numpy_img, (1, 2, 0)))
    plt.show()


# Displays a preview of loaded images
def preview(loader, classes):
    data_iterator = iter(loader)
    images, labels = data_iterator.next()
    show_image(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# Moves a sliding window over an image
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
                    print("Snipping x {} y {} at square size {} \t (scale factor {})"
                          .format(pos[0], pos[1], square_len, 64.0 / square_len))

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


# Transfer-learn the neural network
def train_net(net, loaders, n_epoch, dataset_sizes, cuda=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=.001, momentum=.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

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


# Test the neural network for accuracy
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

    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

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


# Load an image
def image_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def main():
    parser = argparse.ArgumentParser(description='A deep learning powered face recognition app.',
                                     epilog='Written for the INSA Lyon DEEP Project by Anh Pham, Mathilde du Plessix, '
                                            'Romain Latron, Beonît Zhong, Martin Haug. 2018 All rights reserved.')

    parser.add_argument('image', help='Image to recognize faces on')
    parser.add_argument('--epoch', help='Number of epoch', type=int, default=15)
    parser.add_argument('-m', '--model', help='Pass a model file to skip training')
    parser.add_argument('-t', '--train', help='A folder with correctly labeled training data. '
                                              'Will save at model path if option is specified.')
    parser.add_argument('--training-preview', help='Will preview a batch of images from the training set',
                        action='store_true')
    parser.add_argument('--test-preview', help='Will preview a batch of images from the test set', action='store_true')
    parser.add_argument('--outliers', help='Will include the outlier detections in the result as squares',
                        action='store_true')
    parser.add_argument('--color', help='Runs the network on a color version of the image', action='store_true')
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
    else:
        print("On CPU :(")

    datasets = {'train': torchvision.datasets.ImageFolder(root=args.train, transform=transform, loader=image_loader) if args.train is not None else None,
                'val': torchvision.datasets.ImageFolder(root=args.test, transform=transform, loader=image_loader) if args.test is not None else None}

    loaders = {k: torch.utils.data.DataLoader(v, batch_size=4, shuffle=True, num_workers=2) if v is not None else None for (k, v) in datasets.items()}

    if (loaders['train'] is None or loaders['val'] is None) and args.model is None:
        print("You have to specify a model or a training and testing set to use the net")
    elif loaders['train'] is not None and loaders['val'] is not None:
        if args.training_preview:
            preview(loaders['train'], classes)

        net = train_net(net, loaders, args.epoch, {k: len(v) for (k, v) in datasets.items()}, torch.cuda.is_available())

        if args.model is not None:
            torch.save(net.state_dict(), args.model)
            print("Saved model at {}".format(args.model))

    else:
        net.load_state_dict(torch.load(args.model))
        print("Loaded model from {}".format(args.model))

    if args.test is not None:
        if args.test_preview:
            preview(loaders['val'], classes)

        test_net(net, loaders['val'], classes, torch.cuda.is_available())

    orig, bw_img = None, None
    try:
        orig, bw_img = load_single_image(args.image)
    except FileNotFoundError:
        print("Could not open image!")
        exit(-1)

    window_array = image_mover(orig if args.color else bw_img, 0.4, 0.6,
                               terminate_size=max(np.sqrt(orig.size[0] * orig.size[1] * 0.00138), 36))

    fig, ax = plt.subplots(1)
    ax.imshow(orig)
    count = 0
    lowest_score = float("inf")
    highest_score = 0
    with torch.no_grad():
        total_iters = 0
        for window_wrap in window_array:
            crop_real = transform(transforms.functional.crop(orig if args.color else bw_img, window_wrap.y,
                                                             window_wrap.x, window_wrap.height, window_wrap.width))
            if torch.cuda.is_available():
                crop_real = crop_real.to("cuda:0")
            outputs = net(crop_real.unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            if outputs.data[0][1] > 1.2 and predicted[0] == 1:
                window_wrap.is_face(outputs.data[0][1])
                if outputs.data[0][1] > highest_score:
                    highest_score = outputs.data[0][1]

                if outputs.data[0][1] < lowest_score:
                    lowest_score = outputs.data[0][1]

                count += 1
            total_iters += 1
            terminal.print_progress(total_iters,
                                    len(window_array),
                                    prefix='Processing image: ',
                                    suffix='Complete ({} candidates)'.format(count),
                                    bar_length=80)
        print("{} faces detected".format(count))

    med_height = np.median([x.height for x in window_array])
    scan = DBSCAN(eps=med_height * .75, min_samples=2)

    matches = [i for i in window_array if i.face]
    points = np.array([i.get_center() for i in matches])
    clusters = scan.fit(points)

    cl_labels = clusters.labels_

    class_dict = {}

    for i in range(len(matches)):
        if cl_labels[i] not in class_dict:
            class_dict[cl_labels[i]] = []
        class_dict[cl_labels[i]].append(matches[i])

    if -1 in class_dict:
        med = float(np.median([x.score for x in class_dict[-1]]))
        print("Outlier median score {}".format(med))

        circles = []
        for window_wrap in class_dict[-1]:
            if window_wrap.score > med > 1.3:
                circles.append((*(window_wrap.get_center()), window_wrap.height / 2, float(window_wrap.score)))
            elif args.outliers:
                edge = colors.hsv_to_rgb((0, (lowest_score - window_wrap.score) / (lowest_score - highest_score), 1))
                circle_outl = patches.Rectangle(window_wrap.get_top_left(), window_wrap.width, window_wrap.height,
                                                linewidth=2, edgecolor=edge, facecolor='none')
                ax.add_patch(circle_outl)

        overlaps = {}
        for pair in itertools.combinations(circles, 2):
            c1, c2 = pair
            if c1 not in overlaps:
                overlaps[c1] = []
            c1, c2 = pair
            d = np.linalg.norm(np.array([c1[0], c1[1]]) - np.array([c2[0], c2[1]]))
            if (c1[2] - c2[2]) ** 2 <= (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 <= (c1[2] + c2[2]) ** 2\
                    or c1[2] > (d+c2[2]) or c2[2] > (d+c1[2]):
                overlaps[c1].append(c2)

        groups = get_groups(overlaps)

        for group in groups:
            circle = group[0]
            max_score = float("-inf")
            for candidate in group:
                if candidate[3] > max_score:
                    max_score = candidate[3]
                    circle = candidate

            edge = colors.hsv_to_rgb((.15, (lowest_score - circle[3]) / (lowest_score - highest_score), 1))
            circle_outl = patches.Circle((circle[0], circle[1]), (circle[2]),
                                         linewidth=2, edgecolor=edge, facecolor='none')
            ax.add_patch(circle_outl)
        print("Added {} outliers as faces".format(len(groups)))
    for i in range(max(cl_labels) + 1):
        cluster = Cluster(class_dict[i])
        edge = colors.hsv_to_rgb((.38, (lowest_score - cluster.get_max_score()) / (lowest_score - highest_score), 1))
        circle_cluster = patches.Circle(cluster.get_center(), cluster.get_radius(), linewidth=2,
                                        edgecolor=edge, facecolor='none')
        ax.add_patch(circle_cluster)
    plt.show()


if __name__ == "__main__":
    main()
