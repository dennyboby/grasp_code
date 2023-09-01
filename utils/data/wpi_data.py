import glob
import os
from natsort import natsorted

from utils.dataset_processing import grasp, image
from .grasp_data import GraspDatasetBase


class WPIDataset(GraspDatasetBase):
    """
    Dataset wrapper for the WPI dataset.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: WPI Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(WPIDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, 'annotations', '*.txt'))
        self.grasp_files = natsorted(self.grasp_files)
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

        self.depth_files = [f.replace('.txt', '.png').replace('annotations', 'depth_img') for f in self.grasp_files]
        self.rgb_files = [f.replace('depth.png', 'rgb.jpg').replace('depth_img', 'rgb_img') for f in self.depth_files]

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspTriangles.load_from_wpi_file(self.grasp_files[idx])
        gtbbs.scale((self.output_size / 1080, self.output_size / 1920))
        center = self.output_size // 2
        gtbbs.rotate(rot, (center, center))
        gtbbs.zoom(zoom, (center, center))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_file(self.depth_files[idx])
        depth_img.resize((self.output_size, self.output_size))
        depth_img.rotate(rot)
        depth_img.normalise()
        depth_img.zoom(zoom)
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        rgb_img.resize((self.output_size, self.output_size))
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img