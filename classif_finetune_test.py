# -*- encoding: utf-8 -*-

import traceback
import sys
import getopt
import torchvision.transforms as transforms
from model.nn_utils import set_net_train
from utils_dataset import get_images_labels
from utils_image import imread_rgb
from utils_metrics import precision1, mean_avg_precision
from utils_params import *
from utils import *
from classif_finetune import P, labels, test_classif_net, get_embeddings, get_class_net


def usage():
    print('Usage: ' + sys.argv[0] + ' [options]')
    prefix = 'Options:\n\tRequired:\n'
    o1 = ('--dataset=\t<path>\tThe path to the dataset containing all ' +
          'reference images. It should contain a sub-folder "test" ' +
          'containing all test images\n')
    o2 = ('--model=\t<name>\tEither AlexNet or ResNet152 to specify the ' +
          'type of model.\n')
    o3 = ('--weights=\t<file>\tThe filename containing weights of a ' +
          'network trained for sub-region classification.\n')
    o4 = ('--device=\t<int>\tThe GPU device used for testing. ' +
          'If negative, CPU is used.\n')
    o5 = ('--classify=\t<bool>\tTrue/yes/y/1 if the classification ' +
          'feature should be tested. Otherwise, convolutional features ' +
          'are tested.\n')
    o6 = ('--batch=\t<int>\tThe batch size to use.\n')
    o7 = '--help\t\tShow this help\n'
    print(prefix + o1 + o2 + o3 + o4 + o5 + o6 + o7)


def main(dataset_full, model, weights, device, classify, batch_size):
    # training and test sets
    dataset_id = parse_dataset_id(dataset_full)
    match_labels = match_label_functions[dataset_id]
    train_set_full = get_images_labels(dataset_full, match_labels)
    test_set_full = get_images_labels(dataset_full + '/test', match_labels)

    labels_list = [t[1] for t in train_set_full]
    # setup global params so that testing functions work properly
    labels.extend(sorted(list(set(labels_list))))
    P.test_pre_proc = True  # we always pre process images
    P.cuda_device = device
    P.image_input_size = image_sizes[dataset_id]
    P.test_batch_size = batch_size
    P.preload_net = weights
    P.cnn_model = model
    P.feature_size2d = feature_sizes[model, image_sizes[dataset_id]]
    P.embeddings_classify = classify
    out_size = len(labels) if classify else flat_feature_sizes[model, P.image_input_size]
    P.feature_dim = out_size

    print('Loading and transforming train/test sets.')

    # open the images (and transform already if possible)
    # do that only if it fits in memory !
    m, s = read_mean_std(mean_std_files[dataset_id])
    test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(m, s)])
    test_set, test_train_set = [], []
    for im, lab in train_set_full:
        im_o = imread_rgb(im)
        test_train_set.append((test_trans(im_o), lab, im))

    for im, lab in test_set_full:
        if lab not in labels:
            continue
        im_o = imread_rgb(im)
        test_set.append((test_trans(im_o), lab, im))

    print('Testing network on dataset with ID {0}'.format(dataset_id))
    class_net = get_class_net()
    set_net_train(class_net, False)
    c, t = test_classif_net(class_net, test_set)
    print('Classification (TEST): {0} / {1} - acc: {2:.4f}'.format(c, t, float(c) / t))
    test_embeddings = get_embeddings(class_net, test_set, device, out_size)
    ref_embeddings = get_embeddings(class_net, test_train_set, device, out_size)
    sim = torch.mm(test_embeddings, ref_embeddings.t())
    prec1, c, t, _, _ = precision1(sim, test_set, test_train_set)
    mAP = mean_avg_precision(sim, test_set, test_train_set)
    print('Descriptor (TEST): {0} / {1} - acc: {2:.4f} - mAP:{3:.4f}\n'.format(c, t, prec1, mAP))


if __name__ == '__main__':
    options_l = (['help', 'dataset=', 'model=', 'weights=', 'device=',
                 'classify=', 'batch='])
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', options_l)
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    dataset_full, model, weights, device, classify, batch_size = None, None, None, None, None, None
    for opt, arg in opts:
        if opt in ('--help'):
            usage()
            sys.exit()
        elif opt in ('--dataset'):
            dataset_full = check_folder(arg, 'dataset', True, usage)
        elif opt in ('--model'):
            model = check_model(arg, usage)
        elif opt in ('--weights'):
            weights = check_file(arg, 'initialization weights', True, usage)
        elif opt in ('--device'):
            device = check_int(arg, 'device', usage)
        elif opt in ('--classify'):
            classify = check_bool(arg, 'classify', usage)
        elif opt in ('--batch'):
            batch_size = check_int(arg, 'batch', usage)
    if (dataset_full is None or model is None or
            weights is None or device is None or
            classify is None or batch_size is None):
        print('One or more required arguments is missing.')
        usage()
        sys.exit(2)

    with torch.cuda.device(device):
        try:
            main(dataset_full, model, weights, device, classify, batch_size)
        except:
            log_detail(P, None, traceback.format_exc())
            raise
