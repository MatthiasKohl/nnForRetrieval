# -*- encoding: utf-8 -*-

import torchvision.transforms as transforms
from torch.autograd import Variable

from os import path

from utils import *
from model.custom_modules import NormalizeL2Fun
from test_params import P

# TODO possibly transform all images before creating couples/triplets
# when using random transformations, this will require a fixed number of
# transformations per image (need to decide on that number)


def get_device_and_size(net, n):
    # get best device for embeddings, as well as the feature vector size
    # usually, this is the configured cuda device.
    # but it could be CPU if embeddings are too large
    device = P.cuda_device
    out_size = P.siam_feature_dim
    if hasattr(net, 'feature_size'):
        out_size = net.feature_size
    if n * out_size * 4 > 2 ** 30:
        # we will consume more than 1 GB here. use CPU
        device = -1
    return device, out_size


# get all embeddings (feature vectors) of a dataset from a given net
# the net is assumed to be in eval mode
def get_embeddings(net, dataset, device, out_size):
    C, H, W = P.image_input_size
    test_trans = transforms.Compose([])
    if not P.siam_test_pre_proc:
        test_trans.transforms.append(P.siam_test_trans)
    if P.test_norm_per_image:
        test_trans.transforms.append(norm_image_t)

    def batch(last, i, batch):
        embeddings = last
        n = len(batch)
        test_in = tensor(P.cuda_device, n, C, H, W)
        for j in range(n):
            test_in[j] = test_trans(batch[j][0])

        out = net(Variable(test_in, volatile=True))
        for j in range(n):
            embeddings[i + j] = out.data[j]
        return embeddings
    init = tensor(device, len(dataset), out_size)
    return fold_batches(batch, init, dataset, P.siam_test_batch_size)


# accuracy of a net giving feature vectors for each image, evaluated over test set and test ref set (where the images are searched for)
# the model should be in eval mode
# for each pair of images, this only considers the maximal similarity (precision at 1, not the average precision/ranking on the ref set). TODO
def test_descriptor_net(net, testSet, testRefSet, normalized=True):
    d, o = get_device_and_size(net, max(len(testSet), len(testRefSet)))
    test_embeddings = get_embeddings(net, testSet, d, o)
    ref_embeddings = get_embeddings(net, testRefSet, d, o)
    if not normalized:
        test_embeddings = NormalizeL2Fun()(test_embeddings)
        ref_embeddings = NormalizeL2Fun()(ref_embeddings)

    # calculate all similarities as a simple matrix multiplication
    # since inputs are normalized, thus cosine = dot product
    sim = torch.mm(test_embeddings, ref_embeddings.t())
    maxSim, maxIdx = torch.max(sim, 1)
    maxLabel = []
    for i in range(sim.size(0)):
        # get label from ref set which obtained highest score
        maxLabel.append(testRefSet[maxIdx[i, 0]][1])

    # stats
    correct = sum(testLabel == maxLabel[j] for j, (_, testLabel) in enumerate(testSet))
    total = len(testSet)
    sum_pos = sum(sim[i, j] for i, (_, testLabel) in enumerate(testSet) for j, (_, refLabel) in enumerate(testRefSet) if testLabel == refLabel)
    sum_neg = sim.sum() - sum_pos
    sum_max = maxSim.sum()
    lab_dict = dict([(lab, {}) for _, lab in testSet])
    for j, (_, lab) in enumerate(testSet):
        d = lab_dict[lab]
        lab = maxLabel[j]
        d.setdefault(lab, d.get(lab, 0) + 1)
    return correct, total, sum_pos, sum_neg, sum_max, lab_dict


def test_print_siamese(net, testset_tuple, bestScore=0, epoch=0):
    def print_stats(prefix, c, t, avg_pos, avg_neg, avg_max):
        s1 = 'Correct: {0} / {1} -> acc: {2:.4f}\n'.format(c, t, float(c) / t)
        s2 = 'AVG cosine sim (sq dist) values: pos: {0:.4f} ({1:.4f}), neg: {2:.4f} ({3:.4f}), max: {4:.4f} ({5:.4f})'.format(avg_pos, 2 - 2 * avg_pos, avg_neg, 2 - 2 * avg_neg, avg_max, 2 - 2 * avg_max)
        # TODO if not normalized
        log(P.log_file, prefix + s1 + s2)

    testSet, testRefSet = testset_tuple
    net.eval()
    correct, tot, sum_pos, sum_neg, sum_max, lab_dict = test_descriptor_net(net, testSet, testRefSet)
    # can save labels dictionary (predicted labels for all test labels)
    # TODO

    num_pos = sum(testLabel == refLabel for _, testLabel in testSet for _, refLabel in testRefSet)
    num_neg = len(testSet) * len(testRefSet) - num_pos

    if (correct > bestScore):
        bestScore = correct
        prefix = 'SIAM, EPOCH:{0}, SCORE:{1}'.format(epoch, correct)
        P.save_uuid(prefix)
        torch.save(net, path.join(P.save_dir, P.unique_str() + "_best_siam.ckpt"))
    print_stats('TEST - ', correct, tot, sum_pos / num_pos, sum_neg / num_neg, sum_max / len(testSet))
    torch.save(net, path.join(P.save_dir, "model_siam_" + str(epoch) + ".ckpt"))

    # training set accuracy
    trainTestSet = testRefSet[:200]
    correct, tot, sum_pos, sum_neg, sum_max, _ = test_descriptor_net(net, trainTestSet, testRefSet)
    num_pos = sum(testLabel == refLabel for _, testLabel in trainTestSet for _, refLabel in testRefSet)
    num_neg = len(trainTestSet) * len(testRefSet) - num_pos
    print_stats('TRAIN - ', correct, tot, sum_pos / num_pos, sum_neg / num_neg, sum_max / len(trainTestSet))
    net.train()
    return bestScore


def siam_train_stats(net, testset_tuple, epoch, batchCount, last_batch, loss, running_loss, score):
    disp_int = P.siam_loss_int
    test_int = P.siam_test_int
    running_loss += loss
    if batchCount % disp_int == disp_int - 1:
        log(P.log_file, '[{0:d}, {1:5d}] loss: {2:.3f}'.format(epoch + 1, batchCount + 1, running_loss / disp_int))
        running_loss = 0.0
    # test model every x mini-batches
    if ((test_int > 0 and batchCount % test_int == test_int - 1) or
            (last_batch and test_int <= 0)):
        score = test_print_siamese(net, testset_tuple, score, epoch + 1)
    return running_loss, score


def train_siam_couples(net, trainSet, testset_tuple, criterion, optimizer, bestScore=0):
    trans = P.siam_train_trans
    if P.siam_train_pre_proc:
        trans = transforms.Compose([])

    def train_couples(last, i, batch):
        batchCount, score, running_loss = last

        # using sub-batches (only pairs with biggest loss)
        # losses = []
        # TODO

        # get the inputs
        def micro_batch(last, i, batch):
            n = len(batch)
            train_in1 = tensor(P.cuda_device, n, C, H, W)
            train_in2 = tensor(P.cuda_device, n, C, H, W)
            train_labels = tensor(P.cuda_device, n)
            for j, ((im1, im2), lab) in enumerate(batch):
                train_in1[j] = trans(im1)
                train_in2[j] = trans(im2)
                train_labels[j] = lab
            out1, out2 = net(Variable(train_in1), Variable(train_in2))
            loss = criterion(out1, out2, Variable(train_labels))
            loss.backward()
            return last + loss.data[0]

        # zero the parameter gradients, then forward + back prop
        optimizer.zero_grad()
        loss = fold_batches(micro_batch, 0.0, batch, P.siam_train_micro_batch)
        optimizer.step()
        running_loss, score = siam_train_stats(net, testset_tuple, epoch, batchCount, i + len(batch) >= num_train, loss, running_loss, score)
        return batchCount + 1, score, running_loss

    def label_f(i1, l1, i2, l2):
        return 1 if l1 == l2 else -1
    couples = get_couples(trainSet, P.siam_couples_p, label_f)
    num_train = len(couples)
    num_pos = sum(1 for _, lab in couples if lab == 1)
    log(P.log_file, 'training set size:{0}, #pos:{1}, #neg{2}'.format(num_train, num_pos, num_train - num_pos))
    net.train()
    for epoch in range(P.siam_train_epochs):
        random.shuffle(couples)
        init = 0, bestScore, 0.0  # batchCount, bestScore, running_loss
        _, bestScore, _ = fold_batches(train_couples, init, couples, P.siam_train_batch_size)


def train_siam_triplets(net, trainSet, testset_tuple, criterion, optimizer, bestScore=0):
    """
        Train a network
        inputs :
            * trainSet
            * testSet,
            * transformations to apply to image (for train and for test)
            * loss function (criterion)
            * optimizer
    """
    C, H, W = P.image_input_size
    train_trans = P.siam_train_trans
    if P.siam_train_pre_proc:
        train_trans = transforms.Compose([])

    def train_triplets(last, i, batch):
        batchCount, score, running_loss = last

        def micro_batch(last, i, batch):
            n = len(batch)
            train_in1 = tensor(P.cuda_device, n, C, H, W)
            train_in2 = tensor(P.cuda_device, n, C, H, W)
            train_in3 = tensor(P.cuda_device, n, C, H, W)
            # we get a batch of positive couples
            # find random negatives for each couple
            for j, (lab, _, (x1, x2)) in enumerate(batch):
                k = random.randrange(len(trainSet))
                while (trainSet[k][1] == lab):
                    k = random.randrange(len(trainSet))
                train_in1[j] = train_trans(x1)
                train_in2[j] = train_trans(x2)
                train_in3[j] = train_trans(trainSet[k][0])
            out1, out2, out3 = net(Variable(train_in1), Variable(train_in2), Variable(train_in3))
            loss = criterion(out1, out2, out3)
            loss.backward()
            return last + loss.data[0]

        optimizer.zero_grad()
        loss = fold_batches(micro_batch, 0.0, batch, P.siam_train_micro_batch)
        optimizer.step()
        running_loss, score = siam_train_stats(net, testset_tuple, epoch, batchCount, i + len(batch) >= len(couples), loss, running_loss, score)
        return batchCount + 1, score, running_loss

    def train_triplets_hard(last, i, batch):
        batchCount, score, running_loss = last
        # we get a batch of positive couples
        # for each couple, find a negative such that the embedding is
        # semi-hard (using the one with smallest distance can collapse
        # the model, according to Schroff et al - FaceNet) so find
        # one that lies in the margin (alpha) used by the loss to
        # discriminate between positive and negative pair
        # for normalized vectors x and y, we have ||x-y||^2 = 2 - 2xy
        # so finding a negative example such that ||x-x_p||^2 < ||x-x_n||^2
        # is equivalent to having x.x_p > x.x_n
        # after x epochs, we only take the hardest negative examples

        # TODO: triplet selection from Gordo (see mail)

        def micro_batch(last, i, batch):
            n = len(batch)
            train_in1 = tensor(P.cuda_device, n, C, H, W)
            train_in2 = tensor(P.cuda_device, n, C, H, W)
            train_in3 = tensor(P.cuda_device, n, C, H, W)
            for j, (lab, (i1, i2), (x1, x2)) in enumerate(batch):
                em1 = embeddings[i1]
                sqdist_pos = (em1 - embeddings[i2]).pow(2).sum()
                negatives = []
                for k, embedding in enumerate(embeddings):
                    if trainSet[k][1] == lab:
                        continue
                    sqdist_neg = (em1 - embedding).pow(2).sum()
                    if epoch < P.siam_triplets_switch and sqdist_pos >= sqdist_neg:
                        continue
                    negatives.append((k, sqdist_neg))
                if len(negatives) <= 0:
                    p = 'cant find semi-hard neg for'
                    s = 'falling back to random neg'
                    log(P.log_file, '{0} {1}-{2}-{3}, {4}'.format(p, i1, i2, lab, s))
                    k = random.randrange(len(trainSet))
                    while (trainSet[k][1] == lab):
                        k = random.randrange(len(trainSet))
                    x3 = trainSet[k][0]
                else:
                    k3 = min(negatives, key=lambda x: x[1])[0]
                    x3 = trainSet[k3][0]
                train_in1[j] = train_trans(x1)
                train_in2[j] = train_trans(x2)
                train_in3[j] = train_trans(x3)
            out1, out2, out3 = net(Variable(train_in1), Variable(train_in2), Variable(train_in3))
            loss = criterion(out1, out2, out3)
            loss.backward()
            return last + loss.data[0]

        optimizer.zero_grad()
        loss = fold_batches(micro_batch, 0.0, batch, P.siam_train_micro_batch)
        optimizer.step()
        running_loss, score = siam_train_stats(net, testset_tuple, epoch, batchCount, i + len(batch) >= len(couples), loss, running_loss, score)
        return batchCount + 1, score, running_loss

    def shuffle_couples(couples):
        for l in couples:
            random.shuffle(couples[l])
        # get x such that only 20% of labels have more than x couples
        a = np.array([len(couples[l]) for l in couples])
        x = int(np.percentile(a, 80))
        out = []
        keys = couples.keys()
        random.shuffle(keys)
        # append the elements to out in a strided way
        # (up to x elements per label)
        for count in range(x):
            for l in keys:
                if count >= len(couples[l]):
                    continue
                out.append(couples[l][count])
        # the last elements in the longer lists are inserted at random
        for l in keys:
            for i in range(x, len(couples[l])):
                out.insert(random.randrange(len(out)), couples[l][i])
        return out

    # for triplets, only fold over positive couples.
    # then choose negative for each couple specifically
    couples = get_pos_couples(trainSet)
    num_pos = sum(len(couples[l]) for l in couples)
    log(P.log_file, '#pos:{0}'.format(num_pos))
    if P.siam_choice_mode == 'hard':
        f = train_triplets_hard
    else:
        f = train_triplets

    net.train()
    for epoch in range(P.siam_train_epochs):
        # for the 'hard' triplets, we need to know the embeddings of all
        # images at each epoch. so pre-calculate them here
        if P.siam_choice_mode == 'hard':
            net.eval()
            # use the test-train set to obtain embeddings
            # (since it may be transformed differently than train set)
            d, o = get_device_and_size(net, len(testset_tuple[1]))
            embeddings = get_embeddings(net, testset_tuple[1], d, o)
            net.train()

        # for triplets, need to make sure the couples are evenly
        # distributed (such that all batches can have couples from
        # every instance)
        shuffled = shuffle_couples(couples)
        init = 0, bestScore, 0.0  # batchCount, bestScore, running_loss
        _, bestScore, _ = fold_batches(f, init, shuffled, P.siam_train_batch_size)
