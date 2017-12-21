from __future__ import print_function, division

import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.optim as optim
import torch.utils.data
import torchvision
from scipy.misc import imsave
from sklearn.metrics import confusion_matrix as cfmat
from sklearn.metrics import f1_score
from torch.autograd import Variable
from torch.optim import lr_scheduler
from networks import resnet_multi_scale
from RoadSegmentDatasetMultiScale import RoadSegmentDatasetMultiScale

from RoadSegmentDataset import RoadSegmentDataset


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def train_model(model, criterion, optimizer, scheduler, image_names_list, num_epochs=25, model_name=None):
    since = time.time()
    # allocate storage for images
    train_predictions = np.zeros((len(image_names_list["train"]), 400, 400), dtype=np.float32)
    val_predictions = np.zeros((len(image_names_list["val"]), 400, 400), dtype=np.float32)
    test_predictions = np.zeros((len(image_names_list["test"]), 608, 608), dtype=np.float32)

    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        i = 0

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            f1_score_list = []
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            iter = 0
            for data in dataloaders[phase]:
                # get the inputs
                inputs, inputs1, inputs2, labels, image_idx, crop_x_loc_list, crop_y_loc_list = data
                iter = iter + 1

                # manual early stopping
                if False and iter > 2500 and phase == "train":
                    break

                # early saving of test images to understand the progress
                if iter % 200 == 0 and phase == "test":
                    for i in range(test_predictions.shape[0]):
                        test_pred = test_predictions[i, :, :]
                        test_pred = test_pred * 255
                        image_name = image_names_list["test"][i]
                        imsave("./pred_%s/%s_epoch%d.png" % (model_name, image_name, epoch), test_pred)

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    inputs1 = Variable(inputs1.cuda())
                    inputs2 = Variable(inputs2.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, inputs1, inputs2, labels = Variable(inputs), Variable(inputs1), Variable(inputs2), Variable(labels)

                # zero thppe parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model((inputs, inputs1, inputs2))
                # the size of the window we assign for each predict
                pred_step_size = 5 + 1
                max_len = 400 if phase == "train" or phase == "val" else 608
                for idx in range(inputs.size(0)):
                    # calculate the edge cases
                    crop_x_loc = crop_x_loc_list[idx]
                    crop_y_loc = crop_y_loc_list[idx]
                    min_x = max(0, crop_x_loc - int(pred_step_size / 2))
                    max_x = min(max_len, crop_x_loc + int(pred_step_size / 2))
                    min_y = max(0, crop_y_loc - int(pred_step_size / 2))
                    max_y = min(max_len, crop_y_loc + int(pred_step_size / 2))
                    # copy the predictions into arrays so that we can save them as images later
                    if phase == "train":
                        train_predictions[image_idx[idx], min_y:max_y, min_x:max_x] = 1 - softmax(
                            outputs[idx].cpu().data.numpy())[0]
                    elif phase == "val":
                        val_predictions[image_idx[idx], min_y:max_y, min_x:max_x] = 1 - softmax(
                            outputs[idx].cpu().data.numpy())[0]
                    elif phase == "test":
                        test_predictions[image_idx[idx], min_y:max_y, min_x:max_x] = 1 - softmax(
                            outputs[idx].cpu().data.numpy())[0]

                # prediction is the class with the highest output value
                _, preds = torch.max(outputs.data, 1)
                # calculate the loss with prediction and the groundtruth
                loss = criterion(outputs, labels)

                labels_np = labels.cpu().data.numpy()
                preds_np = preds.cpu().numpy()
                # calculate the f1-score for
                f1_score_list.append(f1_score(labels_np, preds_np, average="micro"))
                i = i + 1
                if i % 100 == 0:
                    labels_np = labels.cpu().data.numpy()
                    preds_np = preds.cpu().numpy()
                    print("{} [{}][{}/{}]".format(phase, epoch, i, len(dataloaders[phase])))
                    print("Labels:     {}".format(labels.cpu().data.numpy()))
                    print("Prediction: {}".format(preds.cpu().numpy()))
                    print("Accuracy: %.2f" % torch.mean((preds.cpu() == labels.cpu().data).float()))
                    print("Negative Examples: %.2f" % torch.mean((0 == labels.cpu().data).float()))
                    f1_score_list.append(f1_score(labels_np, preds_np, average="micro"))
                    print("Mean F Score     : %.4f" % np.mean(f1_score_list))
                    # calculate the confusion matrix
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
                image_name = image_names_list["train"][i]
                imsave("./pred_%s/%s_epoch%d.png" % (model_name, image_name, epoch), train_pred)

            for i in range(val_predictions.shape[0]):
                val_pred = val_predictions[i, :, :]
                val_pred = val_pred * 255
                image_name = image_names_list["val"][i]
                imsave("./pred_%s/%s_epoch%d.png" % (model_name, image_name, epoch), val_pred)

            for i in range(test_predictions.shape[0]):
                test_pred = test_predictions[i, :, :]
                test_pred = test_pred * 255
                image_name = image_names_list["test"][i]
                imsave("./pred_%s/%s_epoch%d.png" % (model_name, image_name, epoch), test_pred)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print()

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best=True, filename="%s_%s_checkpoint.pth.tar" % (model_name, epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()

    # initiliaze the hyperparameters
    window_size = [32, 64, 128]
    step_size = 5
    confidence_window = 5
    batch_size = 96 if use_gpu else 16

    # initiliaze the dataset
    data_dir = './'
    image_datasets = {x: RoadSegmentDatasetMultiScale(os.path.join(data_dir, x),
                                                      os.path.join(data_dir, x + "_label"), window_size, step_size,
                                                      confidence_window)
                      for x in ['train', 'val', "test"]}
    image_names_list = {x: image_datasets[x].image_names_list
                        for x in ['train', 'val', "test"]}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=x != "test", num_workers=16)
                   for x in ['test', 'val', 'train']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test', 'val', "train"]}


    # initiliaze the model
    model_conv = resnet_multi_scale(torchvision.models.resnet18(pretrained=True), require_grad=True)

    # UNCOMMENT THIS TO ADD ONE MORE FC LAYER
    # model_conv = resnet_and_fc(model_conv)

    # UNCOMMENT THIS TO USE VGG19 INSTEAD
    # model_conv = torchvision.models.vgg19(pretrained=True)

    # UNCOMMENT if training all layers or using a network from networks.py!
    # for param in model_conv.parameters():
    #    param.requires_grad = False

    # changing the last layer to make binary classification
    print(model_conv)
    if use_gpu:
        model_conv = model_conv.cuda()

    # define the loss function
    if use_gpu:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 2.0], dtype=np.float32)).cuda())
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 2.0], dtype=np.float32)))

    # prepare the learning rate for each layer. Use smaller learning rate for already trained layers and larger for the last fully
    #     connected layer
    lr = 0.0001
    params = []
    for name, value in model_conv.named_parameters():
        if 'bias' in name:
            if 'fc' in name or "last_layer" in name:
                params += [{'params': value, 'lr': 20 * lr, 'weight_decay': 0}]
            else:
                params += [{'params': value, 'lr': 20 * lr, 'weight_decay': 0}]
        else:
            if 'fc' in name:
                params += [{'params': value, 'lr': 5e-1 * lr}]
            else:
                params += [{'params': value, 'lr': 5e-1 * lr}]

    # define the Adam optimizer
    optimizer_conv = optim.Adam(params, weight_decay=0.01)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_name = "full_larger_epoch_multiscale"
    print(model_name)
    # start the training
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, image_names_list, num_epochs=25, model_name=model_name)
