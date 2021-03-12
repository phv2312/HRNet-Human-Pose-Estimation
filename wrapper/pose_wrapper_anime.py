from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import pprint
import cv2

from skimage import measure

"""
Add path
"""
sys.path.append('../lib/')
this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..', 'lib')

if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
##### <- End

import argparse
import os
import pprint
import numpy as np
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from core.inference import get_final_preds, get_max_preds

#import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from utils.transforms import get_affine_transform, flip_back

import dataset
import models


joint_labels_dct = {
    'body_1': 0,
    'nose_1': 1,
    'larm_1' : 2,
    'lelbow_1': 3,
    'lwrist_1' : 4,
    'rarm_1': 5,
    'relbow_1': 6,
    'rwrist_1': 7,
    'lleg_1': 8,
    'lknee_1': 9,
    'lankle_1': 10,
    'rleg_1': 11,
    'rknee_1' : 12,
    'rankle_1': 13,
    'leye_1': 14,
    'reye_1': 15,
    'chin_1': 16,
    'mouth_1': 17,
}
joint_labels = list(joint_labels_dct.keys())

joint_pair = [
    ('neck', 'head'),
    ('body_upper', 'neck'),
    ('neck', 'arm_left'),
    ('neck', 'arm_right'),
    ('body_upper', 'leg_left'),
    ('body_upper', 'leg_right'),

    ('elbow_left', 'wrist_left'),
    ('elbow_right', 'wrist_right'),

    ('elbow_left', 'arm_left'),
    ('elbow_right', 'arm_right'),

    ('ankle_left', 'knee_left'),
    ('ankle_right', 'knee_right'),

    ('leg_left', 'knee_left'),
    ('leg_right', 'knee_right')
]

# dataset dependent configuration for visualization
coco_part_labels = [
    'nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
    'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
    'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r'
]
coco_part_idx = {
    b: a for a, b in enumerate(coco_part_labels)
}
coco_part_orders = [
    ('nose', 'eye_l'), ('eye_l', 'eye_r'), ('eye_r', 'nose'),
    ('eye_l', 'ear_l'), ('eye_r', 'ear_r'), ('ear_l', 'sho_l'),
    ('ear_r', 'sho_r'), ('sho_l', 'sho_r'), ('sho_l', 'hip_l'),
    ('sho_r', 'hip_r'), ('hip_l', 'hip_r'), ('sho_l', 'elb_l'),
    ('elb_l', 'wri_l'), ('sho_r', 'elb_r'), ('elb_r', 'wri_r'),
    ('hip_l', 'kne_l'), ('kne_l', 'ank_l'), ('hip_r', 'kne_r'),
    ('kne_r', 'ank_r')
]

VIS_CONFIG = {
    'COCO': {
        'part_labels': coco_part_labels,
        'part_idx': coco_part_idx,
        'part_orders': coco_part_orders
    }
}

def _update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

def add_joints(image, joints, color, dataset='COCO'):
    part_idx = VIS_CONFIG[dataset]['part_idx']
    part_orders = VIS_CONFIG[dataset]['part_orders']

    def link(a, b, color):
        if part_idx[a] < joints.shape[0] and part_idx[b] < joints.shape[0]:
            jointa = joints[part_idx[a]]
            jointb = joints[part_idx[b]]
            if jointa[2] > 0 and jointb[2] > 0:
                cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                    color,
                    2
                )

    # add joints
    for joint in joints:
        if joint[2] > 0:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 1, color, 2)

    # add link
    for pair in part_orders:
        link(pair[0], pair[1], color)

    return image

def simple_crop_image_by_color(pil_image):
    cv2_im = np.array(pil_image, dtype=np.int16)

    b, r, g = cv2.split(cv2_im)

    sc_im = b + 256 * (g + 1) + 256 * 256 * (r + 1) # single channel
    bg_color = 0 + 256 * (0 + 1) + 256 * 256 * (0 + 1)

    labels = measure.label(sc_im, neighbors=4, background=bg_color)
    for region in measure.regionprops(labels):
        coords = np.asarray(region['coords'])

        if [5,5] in coords:
            print('found')
            _im = region['image'].copy().astype(np.uint8) * 255

            x, y, w, h = cv2.boundingRect(np.stack(np.where(_im == 0)[::-1], axis=-1))
            print (x, y, w, h)

            sub_im = np.array(pil_image, dtype=np.uint8)[y:y+h, x:x+w, :]
            return sub_im

            # cv2.rectangle(_im, (x,y), (x + w, y + h), 127, thickness=10)
            #
            # imgshow(_im)

    return np.asarray(pil_image)

class FakedArugmentPasser:
    def __init__(self, config_path, weight_path):
        self.cfg = config_path
        self.opts = ['TEST.MODEL_FILE', weight_path]

class PoseWrapper:
    def __init__(self, config_path, weight_path, use_gpu = True):
        self.args = FakedArugmentPasser(config_path, weight_path)
        _update_config(cfg, self.args)

        self.logger, self.final_output_dir, self.tb_log_dir = create_logger(
            cfg, self.args.cfg, 'valid'
        )

        self.logger.info(pprint.pformat(self.args))
        self.logger.info(cfg)

        # cudnn related setting
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

        """
        Model & Transforms
        """
        self.model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=False
        )

        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        if use_gpu and not torch.cuda.is_available():
            print ('>>> Enable GPU but <NO GPU FOUND>. Turn into cpu instead ...')
        else:
            print ('>>> Enable GPU')

        self.logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        self.model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

        """
        """
        #self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]

        """
        Heatmap
        """
        #self.parser = HeatmapParser(cfg)

    def _construct_output_image(self, image, joints, color=(255,0,0)):
        output_image = image.copy()

        for person in joints:
            add_joints(output_image, person, color, dataset='COCO')

        return output_image

    def process_single(self, image, use_crop = False):
        """
        Args:
            image: RGB, numpy image

        Returns:

        """
        if use_crop:
            image = simple_crop_image_by_color(image)

        height, width = image.shape[:2]

        with torch.no_grad():
            data_numpy = image #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            c = [width / 2, height / 2]
            s = height / 200.
            r = 0

            c = np.array(c, dtype=np.float)
            s = np.array([s, s], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            # c = c - 1

            trans = get_affine_transform(c, s, r, cfg.MODEL.IMAGE_SIZE)
            input = cv2.warpAffine(
                data_numpy,
                trans,
                tuple(cfg.MODEL.IMAGE_SIZE),
                flags=cv2.INTER_LINEAR
            )
            inp_h, inp_w = input.shape[:2]

            # here

            fixed_coord_Q = np.array([[250, 250, 0]])
            trans_ = np.concatenate([trans, [[0,0,1]]], axis=0)
            fixed_coord_P = fixed_coord_Q @ np.linalg.inv(trans_)

            print ('P:', fixed_coord_P)
            print ('Q:', fixed_coord_Q)

            print ('original image')
            # x = int(fixed_coord_P[0][0])
            # y = int(fixed_coord_P[0][1])
            # cv2.circle(data_numpy, (x,y), radius=3, color=(0,255,0), thickness=2)
            # imgshow(data_numpy)

            print ('image after applying affine transform')
            # x = int(fixed_coord_Q[0][0])
            # y = int(fixed_coord_Q[0][1])
            # cv2.circle(input, (x, y), radius=3, color=(0, 255, 0), thickness=2)
            # imgshow(input)

            print ('image after applying inverse affine transform')
            org_h, org_w = data_numpy.shape[:2]
            #s = np.array([org_w/inp_w, org_h/inp_h])
            inv_trans = get_affine_transform(c, s, r, cfg.MODEL.IMAGE_SIZE, inv=1)
            
            input_tmp = cv2.warpAffine(input, inv_trans, (org_w, org_h), flags=cv2.INTER_LINEAR)
            imgshow(input_tmp)

            debug_im = input.copy()

            input = self.transforms(input).unsqueeze(0).cuda()
            outputs = self.model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            preds, maxvals = get_max_preds(output.clone().cpu().numpy())

            preds   = preds[0]
            maxvals = maxvals[0]

            result_points = {}
            for idx, ((x, y), maxval) in enumerate(zip(preds, maxvals)):

                print ('coor (x,y):', (x, y), ' maxval:', maxval, 'cls:', joint_labels[idx])
                if maxval < 0.25: continue

                x = int(x * 4)
                y = int(y * 4)

                result_points[joint_labels[idx]] = (x,y)

                #
                # cv2.putText(debug_im, joint_labels[idx], (x,y - 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, .4, (255,0,0), thickness=1)
                # cv2.circle(debug_im, (x, y), radius=3, color=(0,255,0), thickness=1)

            # draw single circle point
            for k, point in result_points.items():
                cv2.circle(debug_im, point, radius=2, color=(0, 255, 0), thickness=2)

            # draw pair
            for p1_lbl, p2_lbl in joint_pair:
                if p1_lbl in result_points and p2_lbl in result_points:
                    cv2.line(debug_im, result_points[p1_lbl], result_points[p2_lbl], (255,0,0), thickness=1)

            #imgshow(debug_im)

        return debug_im

import matplotlib.pyplot as plt
import time
def imgshow(im):
    plt.imshow(im)
    plt.show()


if __name__ == '__main__':
    from PIL import Image

    #
    config_path = "/home/kan/Desktop/cinnamon/kp_estimation/keypoint_estimation/experiments/hor01/hrnet/w48_384x288_adam_lr1e-3.yaml"
    weight_path = "/home/kan/Desktop/output_anime_drawing/mpii/pose_hrnet/w48/final_state.pth"

    #
    im_fn = "/home/kan/Desktop/data_kp/anime_drawing/raw_16points/AnimeDrawingsDataset/data/images/16189.jpg"
    im = np.asarray(Image.open(im_fn))

    #
    model = PoseWrapper(config_path, weight_path, use_gpu=True)
    ou = model.process_single(im)

    imgshow(ou)



