import os
import json
import numpy as np
import cv2

"""
l eye_1: 0,
r eye_1: 1,
nose_1 : 2,
mouth_1: 3,
chin_1 : 4
"""

label2id = {
    'l eye_1': 0,
    'r eye_1': 1,
    'nose_1' : 2,
    'mouth_1': 3,
    'chin_1' : 4
}

###
THRESHOLD_AREA = 500

def convert_convention(labeling_data, cv2_im):
    kps = labeling_data['shapes']
    bbox = labeling_data['bounding_box']

    n_joints = len(label2id)
    real_h, real_w = cv2_im.shape[:2]

    # draw bounding boxes
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    x = np.clip(int(x), 0, real_w)
    y = np.clip(int(y), 0, real_h)
    w = np.clip(int(w), 0, real_w)
    h = np.clip(int(h), 0, real_h)

    if w * h >= THRESHOLD_AREA:
        crop_im = cv2_im[y:y+h, x:x+w]

        joints = [[-1, -1] for _ in range(n_joints)]
        joints_vis = [0 for _ in range(n_joints)]

        for kp_dct in kps:
            point = kp_dct['points']
            label = kp_dct['label']

            kp_x, kp_y = point[0]
            kp_x = int(kp_x) - x
            kp_y = int(kp_y) - y

            joints[label2id[label]] = [int(kp_x), int(kp_y)]
            joints_vis[label2id[label]] = 1

        scale  = h / 200.
        center = [w / 2, h / 2]

        info = {
            'scale': scale,
            'center': center,
            'joints': joints,
            'joints_vis': joints_vis,

        }

        return crop_im, info

    return None, None

def process_one_dir(input_image_fn, input_json_fn, output_dir, mode = 'train'):
    #
    out_image_dir = os.path.join(output_dir, 'images')
    out_json_fn  = os.path.join(output_dir, 'annot', "%s.json" % mode)

    #
    basename = os.path.splitext(os.path.basename(input_image_fn))[0]
    cv2_image = cv2.imread(input_image_fn)
    json_data = json.load(open(input_json_fn, 'r', encoding='utf-8'))

    crop_im, annot_info = convert_convention(json_data, cv2_image)
    if crop_im is None or annot_info is None:
        return

    # >>>> image
    out_im_fn = os.path.join(out_image_dir, '%s.png' % basename)
    cv2.imwrite(out_im_fn, crop_im)

    # >>>> annot
    annot_info['image'] = "%s.png" % basename
    old_output_json = json.load(open(out_json_fn, 'r', encoding='utf-8'))
    old_output_json += [annot_info]
    json.dump(old_output_json, open(out_json_fn, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

def process(annot_dir, output_dir):
    dirs = os.listdir(annot_dir)

    for dir in dirs:
        if '.xlsx' in dir: continue
        if not 32 <= int(dir) <= 32: continue

        image_dir = os.path.join(annot_dir, dir, 'images')
        json_dir = os.path.join(annot_dir, dir, 'labels')

        im_fns = os.listdir(image_dir)

        for im_fn in im_fns:
            print('processing image %s ...' % im_fn)

            #
            full_im_fn = os.path.join(image_dir, im_fn)
            full_js_fn = os.path.join(json_dir, os.path.splitext(im_fn)[0] + '.json')

            #
            assert os.path.exists(full_im_fn)
            assert os.path.exists(full_js_fn)

            #
            process_one_dir(full_im_fn, full_js_fn, output_dir, 'valid')

if __name__ == '__main__':
    annot_dir = "/home/kan/Desktop/Cinnamon/pose/labelKeypoint/scripts/processed_hor01"
    output_dir = "."

    process(annot_dir, output_dir)