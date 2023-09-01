import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results, save_results
from utils.dataset_processing import image
from skimage.transform import resize

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str,
                        help='Path to saved network to evaluate')
    parser.add_argument('--rgb_path', type=str, default='data/rgb_img/1_1_0_20230228-163301_rgb.jpg',
                        help='RGB Image path')
    parser.add_argument('--depth_path', type=str, default='data/depth_img/1_1_0_20230228-163301_depth.png',
                        help='Depth Image path')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--save', type=int, default=0,
                        help='Save the results')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args

def numpy_to_torch(s):
    if len(s.shape) == 2:
        return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
    else:
        return torch.from_numpy(s.astype(np.float32))


if __name__ == '__main__':
    args = parse_args()

    # Load image
    logging.info('Loading image...')
    rgb = image.Image.from_file(args.rgb_path)
    rgb.normalise()
    rgb.resize((224, 224))
    rgb.img = rgb.img.transpose((2, 0, 1))
    depth = image.DepthImage.from_file(args.depth_path)
    depth.normalise()
    depth.resize((224, 224))

    x = numpy_to_torch(
        np.expand_dims(
        np.concatenate(
            (np.expand_dims(depth, 0),
                rgb),
            0
        ), 0)
    )

    if args.use_depth and args.use_rgb:
        x = numpy_to_torch(
                np.expand_dims(
                np.concatenate(
                    (np.expand_dims(depth, 0),
                        rgb),
                    0
                ), 0)
            )
        rgb_img = resize(image.Image.from_file(args.rgb_path), (224, 224), preserve_range=True).astype(np.int32)
        depth_img = resize(image.DepthImage.from_file(args.depth_path), (224, 224), preserve_range=True).astype(np.int32)
        g_img = rgb_img
    elif args.use_depth:
        x = numpy_to_torch(np.expand_dims(np.expand_dims(depth, 0), 0))
        rgb_img = None
        depth_img = resize(image.DepthImage.from_file(args.depth_path), (224, 224), preserve_range=True).astype(np.int32)
        g_img = depth_img
    elif args.use_rgb:
        x = numpy_to_torch(np.expand_dims(rgb, 0))
        rgb_img = resize(image.Image.from_file(args.rgb_path), (224, 224), preserve_range=True).astype(np.int32)
        depth_img = None
        g_img = rgb_img

    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network)
    logging.info('Done')

    # Get the compute device
    device = get_device(args.force_cpu)

    with torch.no_grad():
        xc = x.to(device)
        pred = net.predict(xc)

        q_img, ang_img, width_img, length_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'], pred['length'])

        if args.save:
            save_results(
                rgb_img=img_data.get_rgb(rgb, False),
                depth_img=np.squeeze(img_data.get_depth(depth)),
                grasp_q_img=q_img,
                grasp_angle_img=ang_img,
                no_grasps=args.n_grasps,
                grasp_width_img=width_img,
                grasp_length_img=length_img
            )
        else:
            fig = plt.figure(figsize=(10, 10))
            plot_results(fig=fig,
                         rgb_img=rgb_img,
                         grasp_q_img=q_img,
                         grasp_angle_img=ang_img,
                         depth_img=depth_img,
                         no_grasps=args.n_grasps,
                         grasp_width_img=width_img,
                         grasp_length_img=length_img,
                         grasp_img=g_img)
            fig.savefig('img_result.pdf')
