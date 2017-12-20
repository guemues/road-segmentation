from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from RoadSegmentDataset import RoadSegmentDataset
from sklearn.metrics import confusion_matrix as cfmat
from sklearn.metrics import f1_score
import torch.optim
import torch.utils.data
from scipy.misc import imsave

plt.ion()  # interactive mode

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    train_predictions = np.zeros((100, 400, 400), dtype=np.float32)
    val_predictions = np.zeros((10, 400, 400),dtype=np.float32)
    test_predictions = np.zeros((50, 608, 608),dtype=np.float32)

    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        i = 0

        # Each epoch has a training and validation phase
        for phase in ['test', 'val', 'train']:
            f1_score_list = np.array([])
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels, image_idx, crop_x_loc_list, crop_y_loc_list = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                pred_step_size = 5
                max_len = 400 if phase=="train" or phase=="val" else 608
                for idx in range(inputs.size(0)):
                    crop_x_loc = crop_x_loc_list[idx]
                    crop_y_loc = crop_y_loc_list[idx]
                    min_x = max(0, crop_x_loc - int(pred_step_size / 2))
                    max_x = min(max_len, crop_x_loc + int(pred_step_size / 2))
                    min_y = max(0, crop_y_loc - int(pred_step_size / 2))
                    max_y = min(max_len, crop_y_loc + int(pred_step_size / 2))
                    if phase == "train":
                        train_predictions[image_idx[idx], min_y:max_y, min_x:max_x] = softmax(
                            outputs[idx].cpu().data.numpy())[0]
                    elif phase == "val":
                        val_predictions[image_idx[idx], min_y:max_y, min_x:max_x] = softmax(
                            outputs[idx].cpu().data.numpy())[0]
                    elif phase =="test":
                        test_predictions[image_idx[idx], min_y:max_y, min_x:max_x] = softmax(
                            outputs[idx].cpu().data.numpy())[0]

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                i = i + 1
                if i % 100 == 0:
                    labels_np = labels.cpu().data.numpy()
                    preds_np = preds.cpu().numpy()
                    print("{} [{}][{}/{}]".format(phase, epoch, i, len(dataloaders[phase])))
                    print("Labels:     {}".format(labels.cpu().data.numpy()))
                    print("Prediction: {}".format(preds.cpu().numpy()))
                    print("Accuracy: %.2f" % torch.mean((preds.cpu() == labels.cpu().data).float()))
                    # print("Recall  : %.2f " % (np.sum(np.logical_and(preds_np, labels_np))/np.sum(labels_np)))
                    print("Negative Examples: %.2f" % torch.mean((0 == labels.cpu().data).float()))
                    f1_score_list = np.append(f1_score_list, f1_score(labels_np, preds_np, average="micro"))
                    if f1_score_list.shape[0] > 100:
                        f1_score_list = f1_score_list[-100]
                    print("Mean F Score     : %.4f" % np.mean(f1_score_list))
                    print(cfmat(labels_np, preds_np))

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                # print("%.4f" % loss.data[0])
                running_corrects += torch.sum(preds == labels.data)

            print("Saving prediction images")
            for i in range(train_predictions.shape[0]):
                train_pred = train_predictions[i, :, :]
                train_pred = train_pred * 255
                imsave("./pred/train_pred%i_epoch%d.png"%(i,epoch), train_pred)

            for i in range(val_predictions.shape[0]):
                val_pred = val_predictions[i, :, :]
                val_pred = val_pred * 255
                imsave("./pred/val_pred%i_epoch_%d.png"%(i,epoch), val_pred)

            for i in range(test_predictions.shape[0]):
                test_pred = test_predictions[i, :, :]
                test_pred = test_pred * 255
                imsave("./pred/test_pred%i_epoch_%d.png"%(i,epoch), test_pred)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    # Data augmentation and normalization for training
    # Just normalization for validation
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    """
    use_gpu = torch.cuda.is_available()
    window_size = 40
    step_size = 5
    confidence_window = 5
    batch_size = 128 if use_gpu else 16

    data_dir = './'
    image_datasets = {x: RoadSegmentDataset(os.path.join(data_dir, x),
                                            os.path.join(data_dir, x + "_label"), window_size, step_size,
                                            confidence_window)
                      for x in ['train', 'val', "test"]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=(x=="train"), num_workers=4)
                   for x in ['test', 'val', 'train']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test', 'val', "train"]}
    class_names = ["Not Road", "Road"]


    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.25, 0.25, 0.25])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


    """
    for i in range(1):
        # Get a batch of training data
        inputs, classes = next(iter(dataloaders['train']))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        imshow(out, title=[class_names[x] for x in classes])
        plt.show()
    """
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    print(model_conv)
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    if use_gpu:
        model_conv = model_conv.cuda()

    if use_gpu:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 2.0], dtype=np.float32)).cuda())
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 2.0], dtype=np.float32)))

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=25)
