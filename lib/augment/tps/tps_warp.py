import os, sys
sys.path.append(os.path.dirname(__file__))
import numpy as np
import cv2
from warp_image import warp_images
from augment.core import CoreTransform
import matplotlib.pyplot as plt
def imshow(im):
    plt.imshow(im)
    plt.show()

def _get_regular_grid(image, points_per_dim):
    nrows, ncols = image.shape[0], image.shape[1]
    rows = np.linspace(0, nrows, points_per_dim)
    cols = np.linspace(0, ncols, points_per_dim)
    rows, cols = np.meshgrid(rows, cols)
    return np.dstack([cols.flat, rows.flat])[0]


def _generate_random_vectors(image, src_points, scale):
    dst_pts = src_points + np.random.uniform(-scale, scale, src_points.shape)
    return dst_pts

def _generate_fix_coordinate(input_image, points_per_dim, scale=None):
    """
    Generate the source * target points for TPS algorithm
    Args:
        input_image:
        points_per_dim:
        scale:

    Returns:

    """
    h, w = input_image.shape[:2]
    scale = scale if scale is not None else 0.1 * w
    src_points = _get_regular_grid(input_image, points_per_dim)
    tgt_points = _generate_random_vectors(input_image, src_points, scale)

    return src_points, tgt_points

def _thin_plate_spline_warp(image, src_points, dst_points, keep_corners=True, interpolation_order=1):
    width, height = image.shape[:2]
    if keep_corners:
        corner_points = np.array(
            [[0, 0], [0, width], [height, 0], [height, width]])
        src_points = np.concatenate((src_points, corner_points))
        dst_points = np.concatenate((dst_points, corner_points))
    out = warp_images(src_points, dst_points,
                      np.moveaxis(image, 2, 0),
                      (0, 0, width - 1, height - 1), interpolation_order=interpolation_order)
    return np.moveaxis(np.array(out), 0, 2)


def tps_warp(image, points_per_dim, scale):
    width, height = image.shape[:2]
    src = _get_regular_grid(image, points_per_dim=points_per_dim)
    dst = _generate_random_vectors(image, src, scale=scale*width)
    out = _thin_plate_spline_warp(image, src, dst)
    return out

def tps_warp_2(image, dst, src, interpolation_order=1):
    out = _thin_plate_spline_warp(image, src, dst, interpolation_order=interpolation_order)
    return out

def find_correspondence_vertex(point_loc, org_mask, dst_mask):
    x, y = point_loc
    x = int(x); y = int(y)

    org_b, org_g, org_r = org_mask[:,:,0], org_mask[:,:,1], org_mask[:,:,2]
    org_single_mask = org_b.astype(np.int64) * 255 * 255 + org_g.astype(np.int64) * 255 + org_r.astype(np.int64)

    dst_b, dst_g, dst_r = dst_mask[:,:,0], dst_mask[:,:,1], dst_mask[:,:,2]
    dst_single_mask = dst_b.astype(np.int64) * 255 * 255 + dst_g.astype(np.int64) * 255 + dst_r.astype(np.int64)

    org_color = org_single_mask[y,x]
    new_ys, new_xs =  np.where(dst_single_mask == org_color)

    n_point = len(new_ys)
    if n_point <= 0:
        print ('can not find correspondence key-points for loc.(x-%d,y-%d)' % (x, y))
        return tuple([-1., -1.])
    else:
        new_point_loc = tuple([new_xs[n_point//2], new_ys[n_point//2]])
        return new_point_loc

class TPSTransform(CoreTransform):
    def __init__(self, version):
        super(TPSTransform, self).__init__()
        self.version = version

    def set_random_parameters(self, input_image, points_per_dim=3, scale_factor=0.1, **kwargs):
        h, w = input_image.shape[:2]
        src_points = _get_regular_grid(input_image, points_per_dim)
        dst_points = _generate_random_vectors(input_image, src_points, scale=scale_factor * w)
        arugments = {
            'points_per_dim': points_per_dim,
            'scale': scale_factor * w,
            'src_points': src_points,
            'dst_points': dst_points
        }

        self.params = arugments

    def get_random_parameters(self):
        return self.params

    def transform_coordinate(self, xy_coords, input_image, output_size, interpolation_mode, **kwargs):
        # build image mask
        h, w = input_image.shape[:2]
        mask = np.zeros(shape=(h,w,3), dtype=np.uint8)

        # draw mask
        g_id = 1
        for x, y in xy_coords:
            x = int(x)
            y = int(y)
            mask[y, x] = [g_id, g_id, g_id]
            cv2.circle(mask, (x,y), radius=2, color=(g_id, g_id, g_id), thickness=2)

            g_id += 1

        out_mask = self.transform_image(mask, output_size, 'nearest')
        #imshow(out_mask)

        #
        output_points = []
        for x, y in xy_coords:
            new_x, new_y = find_correspondence_vertex(point_loc=(x,y), org_mask=mask, dst_mask=out_mask)
            output_points += [(new_x, new_y)]

        return np.array(output_points)

    def transform_image(self, input_image, output_size, interpolation_mode, **kwargs):
        tps_params = self.params
        src_points = tps_params['src_points']
        dst_points = tps_params['dst_points']

        if interpolation_mode == 'linear':
            i_order = 1
            i_resize_type = cv2.INTER_CUBIC
        elif interpolation_mode == 'nearest':
            i_order = 0
            i_resize_type = cv2.INTER_NEAREST
        else:
            raise Exception('not support for interpolation mode: %s' % str(interpolation_mode))

        out = tps_warp_2(input_image, dst_points, src_points, interpolation_order=i_order)

        in_h, in_w = input_image.shape[:2]
        ou_w, ou_h = output_size

        ratio_x = in_w / ou_w
        ratio_y = in_h / ou_h
        ratio_max = max([ratio_x, ratio_y])
        scale_factor = 1. / ratio_max

        ou_est_img = cv2.resize(out, interpolation=i_resize_type,
                                dsize=(int(scale_factor * in_w), int(scale_factor * in_h)))
        ou_est_h, ou_est_w = ou_est_img.shape[:2]

        pad_h = ou_h - ou_est_h
        pad_w = ou_w - ou_est_w

        if pad_h > 0:
            ou_est_img = cv2.copyMakeBorder(ou_est_img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=0)

        if pad_w > 0:
            ou_est_img = cv2.copyMakeBorder(ou_est_img, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        #
        return ou_est_img


if __name__ == '__main__':
    import cv2
    import numpy as np

    image_path = "/home/kan/Desktop/cinnamon/CharacterGAN/datasets/hor02_037_C/C/output/C0001.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    output_size = (512, 768)
    fixed_src_points = np.array([[100, 75], [80, 108]]) # xy coordinates

    #
    tps_transform = TPSTransform(version='tps')
    tps_transform.set_random_parameters(input_image=image, points_per_dim=3, scale_factor=0.1)

    augment_image = tps_transform.transform_image(input_image=image, output_size=output_size, interpolation_mode='linear')
    print ('original,', image.shape)
    imshow(image)

    print ('augment,', augment_image.shape)
    imshow(augment_image)

    augment_keypoints = tps_transform.transform_coordinate(xy_coords=fixed_src_points, input_image=image, output_size=output_size, interpolation_mode='nearest')
    print (fixed_src_points)
    print (augment_keypoints)
    for point in fixed_src_points:
        x, y = point
        cv2.circle(image, (x,y), radius=2, color=(255,0,0), thickness=2)
    imshow(image)

    for point in augment_keypoints:
        x, y = point
        cv2.circle(augment_image, (x, y), radius=2, color=(255, 0, 0), thickness=2)
    imshow(augment_image)
