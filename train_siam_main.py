# -*- encoding: utf-8 -*-

import torchvision.models as models
import torch.optim as optim
from PIL import Image

from os import path

from model.siamese import *
from train_classif import *
from train_siam import *
from dataset import ReadImages
from test_params import P


def main():

    def match(x):
        return x.split('/')[-1].split('-')[0]
    # training and test sets (scaled to 300 on the small side)
    trainSetFull = ReadImages.readImageswithPattern(
        P.dataset_full, match)
    testSetFull = ReadImages.readImageswithPattern(
        P.dataset_full + '/test', match)

    # define the labels list
    listLabel = [t[1] for t in trainSetFull if 'wall' not in t[1]]
    labels = list(set(listLabel))  # we have to give a number for each label

    print('Loading and transforming train/test sets.')

    # open the images (and transform already if possible)
    # do that only if it fits in memory !
    def trans_dataset(dataset, pre_procs, trans, filters=None):
        if filters is None:
            filters = [lambda x, y: True for _ in pre_procs]
        outs = [[] for _ in pre_procs]
        for im, lab in dataset:
            im_o = Image.open(im)
            for out, pre_proc, t, f in zip(outs, pre_procs, trans, filters):
                if f(im, lab):
                    im_out = t(im_o) if pre_proc else im_o
                    out.append((im_out, lab))
        return outs

    train_pre_procs = [P.classif_train_pre_proc, P.classif_test_pre_proc, P.siam_train_pre_proc, P.siam_test_pre_proc]
    train_trans = [P.classif_train_trans, P.classif_test_trans, P.siam_train_trans, P.siam_test_trans]

    trainSetClassif, testTrainSetClassif, trainSetSiam, testTrainSetSiam = trans_dataset(trainSetFull, train_pre_procs, train_trans)

    test_pre_procs = [P.classif_test_pre_proc, P.siam_test_pre_proc]
    test_trans = [P.classif_test_trans, P.siam_test_trans]
    test_filters = [lambda im, lab: lab in labels for _ in test_trans]
    testSetClassif, testSetSiam = trans_dataset(testSetFull, test_pre_procs, test_trans, test_filters)

    print('Starting classification training')

    if P.finetuning:
        class_net = TuneClassif(models.resnet152(pretrained=True), len(labels), untrained_blocks=P.untrained_blocks)
    else:
        class_net = models.resnet152()

    # class_net = torch.load(path.join(P.save_dir, 'best_classif_1.ckpt'))

    if P.cuda_device >= 0:
        class_net.cuda()
    else:
        class_net.cpu()
    class_net.train()
    optimizer = optim.SGD((p for p in class_net.parameters() if p.requires_grad), lr=P.classif_lr, momentum=P.classif_momentum, weight_decay=P.classif_weight_decay)
    criterion = nn.CrossEntropyLoss()
    testset_tuple = (testTrainSetClassif, testSetClassif)
    # score = test_print_classif(class_net, testset_tuple, labels)
    score = 0
    # TODO try normal weight initialization in classification training (see faster rcnn in pytorch)
    train_classif(class_net, trainSetClassif, testset_tuple, labels, criterion, optimizer, bestScore=score)

    print('Finished classification training')
    print('Starting descriptor training')

    # for ResNet152, spatial feature dimensions are 8x8 (for 227x227 input)
    # for AlexNet, it's 6x6 (for 227x227 input)
    net = Siamese1(class_net, feature_dim=P.siam_feature_dim, feature_size2d=P.siam_feature_out_size2d)
    if P.cuda_device >= 0:
        net.cuda()
    else:
        net.cpu()
    net.train()

    optimizer = optim.SGD((p for p in net.parameters() if p.requires_grad), lr=P.siam_lr, momentum=P.siam_momentum, weight_decay=P.siam_weight_decay)
    # criterion = nn.CosineEmbeddingLoss(margin=P.siam_cos_margin, size_average=P.siam_loss_avg)
    if P.siam_train_mode == 'couples':
        criterion = nn.CosineEmbeddingLoss(margin=P.siam_cos_margin, size_average=P.siam_loss_avg)
    else:
        criterion = TripletLoss(margin=P.siam_triplet_margin, size_average=P.siam_loss_avg)
    testset_tuple = (testSetSiam, testTrainSetSiam)
    # score = test_print_siamese(net, testset_tuple, P.siam_test_batch_size)
    score = 0
    if P.siam_train_mode == 'couples':
        f = train_siam_couples
    else:
        f = train_siam_triplets
    f(net, trainSetSiam, testset_tuple, criterion, optimizer, bestScore=score)

    print('Finished descriptor training')


if __name__ == '__main__':
    with torch.cuda.device(P.cuda_device):
        main()
