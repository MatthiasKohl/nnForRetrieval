# -*- encoding: utf-8 -*-

import torchvision.transforms as transforms
from torch.autograd import Variable

from os import path

from utils import *
from test_params import P


# test a classifier model. it should be in eval mode
def test_classif_net(net, testSet, labels, batchSize):
    """
        Test the network accuracy on a testSet
        Return the number of succes and the number of evaluations done
    """
    trans = transforms.Compose([])
    if not P.classif_test_pre_proc:
        trans.transforms.append(P.classif_test_trans)
    if P.test_norm_per_image:
        trans.transforms.append(norm_image_t)

    def eval_batch_test(last, i, is_final, batch):
        correct, total = last
        C, H, W = P.image_input_size
        test_in = tensor(P.cuda_device, len(batch), C, H, W)
        for j, (testIm, _) in enumerate(batch):
            test_in[j] = trans(testIm)

        out = net(Variable(test_in, volatile=True)).data
        _, predicted = torch.max(out, 1)
        total += len(batch)
        correct += sum(labels.index(testLabel) == predicted[j][0] for j, (_, testLabel) in enumerate(batch))
        return correct, total

    return fold_batches(eval_batch_test, (0, 0), testSet, batchSize)


def test_print_classif(net, testset_tuple, labels, bestScore=0, epoch=0):
    testTrainSet, testSet = testset_tuple
    net.eval()
    c, t = test_classif_net(net, testSet, labels, P.classif_test_batch_size)
    if (c > bestScore):
        bestScore = c
        prefix = 'CLASSIF, EPOCH:{0}, SCORE:{1}'.format(epoch, c)
        P.save_uuid(prefix)
        torch.save(net, path.join(P.save_dir, P.unique_str() + "_best_classif.ckpt"))
    P.log('TEST - correct:{0}/{1} - acc:{2}'.format(c, t, float(c) / t))

    c, t = test_classif_net(net, testTrainSet, labels, P.classif_test_batch_size)
    torch.save(net, path.join(P.save_dir, "model_classif_" + str(epoch) + ".ckpt"))
    P.log("TRAIN - correct:{0}/{1} - acc:{2}".format(c, t, float(c) / t))
    net.train()
    return bestScore


def train_classif(net, trainSet, testset_tuple, labels, criterion, optimizer, bestScore=0):
    """
        Train a network
        inputs :
            * trainSet
            * testSet,
            * transformations to apply to image (for train)
            * loss function (criterion)
            * optimizer
    """
    def train_batch(last, i, is_final, batch):
        batchCount, score, running_loss = last
        batchSize = len(batch)
        # get the inputs
        C, H, W = P.image_input_size
        train_in = tensor(P.cuda_device, batchSize, C, H, W)
        for j in range(batchSize):
            if P.classif_train_pre_proc:
                train_in[j] = batch[j][0]
            else:
                train_in[j] = P.classif_train_trans(batch[j][0])

        # get the labels
        lab = tensor_t(torch.LongTensor, P.cuda_device, batchSize)
        for j in range(batchSize):
            lab[j] = labels.index(batch[j][1])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        out = net(Variable(train_in))
        loss = criterion(out, Variable(lab))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        disp_int = P.classif_loss_int
        if batchCount % disp_int == disp_int - 1:
            P.log('[{0:d}, {1:5d}] loss: {2:.3f}'.format(epoch + 1, batchCount + 1, running_loss / disp_int))
            running_loss = 0.0

        test_int = P.classif_test_int
        if ((test_int > 0 and batchCount % test_int == test_int - 1) or
                (test_int <= 0 and is_final)):
            score = test_print_classif(net, testset_tuple, labels, score, epoch + 1)
        return batchCount + 1, score, running_loss

    net.train()
    for epoch in range(P.classif_train_epochs):
        # annealing
        optimizer = anneal(net, optimizer, epoch, P.classif_annealing)

        init = 0, bestScore, 0.0  # batch count, score, running loss
        random.shuffle(trainSet)
        _, bestScore, _ = fold_batches(train_batch, init, trainSet, P.classif_train_batch_size)
