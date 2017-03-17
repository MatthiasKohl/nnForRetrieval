# -*- encoding: utf-8 -*-

import tempfile
import torchvision.transforms as transforms
import torchvision.models as models
from datetime import datetime
from os import rename, path
from utils import *


class TestParams(object):

    def __init__(self):

        def match_fou_clean2(x):
            s = x.split('/')[-1].split('_')
            return s[0] + s[1]

        def match_video(x):
            return x.split('/')[-1].split('-')[0]

        # UUID for these parameters (current time)
        self.uuid = datetime.now()

        # general parameters
        self.dataset_full = 'data/pre_proc/CLICIDE_max_224sq'
        self.dataset_name = self.dataset_full.split('/')[-1].split('_')[0]
        self.mean_std_file = 'data/fourviere_224sq_train_ms.txt' if self.dataset_name == 'fourviere' else 'data/CLICIDE_224sq_train_ms.txt'
        self.dataset_match_img = match_fou_clean2 if self.dataset_name == 'fourviere' else match_video
        self.finetuning = True
        self.cnn_model = models.alexnet
        self.feature_size2d = (6, 6)
        self.image_input_size = (3, 224, 224)
        self.save_dir = 'data'
        self.log_file = path.join(self.save_dir, self.unique_str() + '.log')
        self.cuda_device = 1
        self.test_norm_per_image = False

        # in ResNet, before first layer, there are 2 modules with parameters.
        # then number of blocks per layers:
        # ResNet152 - layer 1: 3, layer 2: 8, layer 3: 36, layer 4: 3
        # ResNet50 - layer 1: 3, layer 2: 4, layer 3: 6, layer 4: 3
        # finally, a single FC layer is used as classifier
        # in AlexNet, there are 5 convolutional layers with parameters
        # and 3 FC layers in the classifier
        self.untrained_blocks = (2 + 3 + 8 + 36) if self.cnn_model is models.resnet152 else 4

        # read mean and standard of dataset here to define transforms already
        m, s = readMeanStd(self.mean_std_file)

        # Classification net general and test params
        self.classif_preload_net = ''
        self.classif_test_upfront = False
        self.classif_train = False
        self.classif_test_batch_size = 128
        self.classif_test_pre_proc = True
        self.classif_test_trans = transforms.Compose([transforms.ToTensor()])
        if not self.test_norm_per_image:
            # normalization not done per image during test
            self.classif_test_trans.transforms.append(transforms.Normalize(m, s))

        # Classification net training params
        self.classif_train_epochs = 50
        self.classif_train_batch_size = 32
        self.classif_train_pre_proc = False
        self.classif_train_aug_rot = r = 45
        self.classif_train_aug_hrange = hr = 0.2
        self.classif_train_aug_vrange = vr = 0.2
        self.classif_train_aug_hsrange = hsr = 0.2
        self.classif_train_aug_vsrange = vsr = 0.2
        self.classif_train_trans = transforms.Compose([random_affine(rotation=r, h_range=hr, v_range=vr, hs_range=hsr, vs_range=vsr), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(m, s)])
        self.classif_lr = 1e-2
        self.classif_momentum = 0.9
        self.classif_weight_decay = 5e-4
        self.classif_optim = 'SGD'
        self.classif_annealing = {30: 0.1}
        self.classif_loss_int = 10
        self.classif_test_int = 0

        # settings for feature net constructed from classification net
        self.feature_net_average = False
        self.feature_net_classify = False
        self.feature_net_upfront = False

        # Siamese net general and testing params
        self.siam_preload_net = ''
        self.siam_test_upfront = True
        self.siam_train = True
        self.siam_feature_dim = 4096
        self.siam_cos_margin = 0  # 0: pi/2 angle, 0.5: pi/3, sqrt(3)/2: pi/6
        self.siam_loss_avg = False
        self.siam_test_batch_size = 32
        self.siam_test_pre_proc = True
        self.siam_test_trans = transforms.Compose([transforms.ToTensor()])
        if not self.test_norm_per_image:
            # normalization not done per image during test
            self.siam_test_trans.transforms.append(transforms.Normalize(m, s))

        # Siamese net training params
        # for train mode: 'couples': using cosine loss
        # 'triplets': using triplet loss
        # choice mode: for 'couples':
        # 'rand': using random couples
        # 'hard': using all positives and hardest negative couples
        # for 'triplets':
        # 'rand': using random negatives for all positives
        # 'semi-hard':semi-hard triplets
        # (see FaceNet paper by Schroff et al)
        self.siam_train_mode = 'couples'
        self.siam_choice_mode = 'rand'
        self.siam_triplet_margin = 0.2
        # for triplet:semi-hard mode, number of epochs after which we
        # take only the hardest examples:
        self.siam_triplets_switch = 10
        self.siam_train_trans = self.classif_train_trans
        self.siam_train_pre_proc = False
        self.siam_couples_p = 0.9
        self.siam_train_batch_size = 256
        self.siam_train_micro_batch = 32
        self.siam_lr = 1e-3
        self.siam_momentum = 0.9
        self.siam_weight_decay = 0.0
        self.siam_optim = 'SGD'
        self.siam_annealing = {}
        self.siam_train_epochs = 50
        self.siam_loss_int = 10
        self.siam_test_int = 50

    def unique_str(self):
        return self.uuid.strftime('%Y%m%d-%H%M%S-%f')

    def save(self, f, prefix):
        f.write('{0}\n'.format(prefix))
        for name, value in sorted(vars(self).items()):
            if name == 'uuid':
                continue
            if name in ('classif_test_trans', 'classif_train_trans', 'siam_test_trans', 'siam_train_trans'):
                value = trans_str(value)
            elif name in ('cnn_model', 'dataset_match_img'):
                value = fun_str(value)
            f.write('{0}:{1}\n'.format(name, value))
        f.close()

    def save_uuid(self, prefix):
        f = tempfile.NamedTemporaryFile(dir=self.save_dir, delete=False)
        self.save(f, prefix)
        # the following will not work on Windows (would need to add a remove first)
        rename(f.name, path.join(self.save_dir, self.unique_str() + '.params'))


# global test params:
P = TestParams()
