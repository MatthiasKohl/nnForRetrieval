# -*- encoding: utf-8 -*-

from utils_image import *
from utils_params import *
from utils import *


# parameters for the sub-regions classification training with AlexNet
class Params(object):

    def __init__(self):

        # UUID for these parameters (current time)
        self.uuid = datetime.now()
        self.log_file = path.join(self.save_dir, self.unique_str() + '.log')

        # general parameters
        self.dataset_full = 'data/pre_proc/fourviere_clean2_384'
        self.cuda_device = 0
        self.save_dir = 'data'
        self.dataset_id = self.dataset_full.split('/')[-1]
        self.mean_std_file = mean_std_files[self.dataset_id]
        self.match_labels_f = match_image[self.dataset_id]
        self.image_input_size = image_sizes[self.dataset_id]
        self.num_classes = num_classes[self.dataset_id]
        self.feature_size2d = feature_sizes[('alexnet', self.image_input_size)]

        # in AlexNet, there are 5 convolutional layers with parameters
        # and 3 FC layers in the classifier
        self.untrained_blocks = 4

        # read mean and standard of dataset here to define transforms already
        m, s = readMeanStd(self.mean_std_file)

        # Classification net general and test params
        self.preload_net = ''  # allows to continue training a network
        self.bn_model = ''
        self.test_upfront = False
        self.train = False
        self.test_pre_proc = True
        self.test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(m, s)])

        # Classification net training params
        self.train_epochs = 50
        self.train_batch_size = 32
        self.train_micro_batch = 1
        self.train_aug_rot = r = 180
        self.train_aug_hrange = hr = 0
        self.train_aug_vrange = vr = 0
        self.train_aug_hsrange = hsr = 0.5
        self.train_aug_vsrange = vsr = 0.5
        self.train_aug_hflip = hflip = True
        trans = transforms.Compose([random_affine_noisy_cv(rotation=r, h_range=hr, v_range=vr, hs_range=hsr, vs_range=vsr, h_flip=hflip), transforms.ToTensor(), transforms.Normalize(m, s)])
        # list of transforms for all scales
        # the train_trans parameter should be a list of same
        # length representing the train transformation for each scale
        self.train_sub_scales = [transforms.Compose([]), transforms.Compose([scale_cv(224)])]
        # transformation for each scale
        self.train_trans = [trans, trans]
        self.train_pre_proc = [False, False]

        self.train_lr = 1e-3
        self.train_momentum = 0.9
        self.train_weight_decay = 5e-4
        self.train_optim = 'SGD'
        self.train_annealing = {30: 0.1}
        self.train_loss_avg = True
        self.train_loss_int = 10
        self.train_test_int = 0
        # the batch norm layer cannot be trained if the micro-batch size
        # is too small, as global variances/means cannot be properly
        # approximated in this case. so train only when having a batch
        # of at least 16
        self.train_bn = self.train_micro_batch >= 16 or (self.train_micro_batch <= 0 and (self.train_batch_size >= 16 or self.train_batch_size <= 0))

        # Descriptor net parameters
        # if True, test the network as a descriptor
        # (using the normalized classification output):
        self.test_descriptor_net = False
        # the threshold (in Bytes) for embeddings to be computed on GPU
        self.embeddings_cuda_size = 2 ** 30


# global test params:
P = Params()
