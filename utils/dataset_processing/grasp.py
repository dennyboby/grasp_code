import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon, disk
from skimage.feature import peak_local_max

IMAGE_SHAPE = (224, 224)
DISK_RADIUS = 3

def _gr_text_to_no(l, offset=(0, 0)):
    """
    Transform a single point from a WPI file line to a pair of ints.
    :param l: Line from WPI grasp file (str)
    :param offset: Offset to apply to point positions
    :return: Point [row, col]
    """
    col, row = l.split()
    return [int(round(float(row))) - offset[0], int(round(float(col))) - offset[1]]

def img_to_cord(points):
    """
    Transform points from image pixel cord (row, col) to x, y cord axis
    :param points: array of points in format (row, col) of image axis
    :return: Points in format of [x, y] cord axis
    """
    # convert row, col to col, row
    points = points[:, ::-1]
    new_points = np.array([0, IMAGE_SHAPE[1]]) - (points * np.array([-1, 1]))
    return new_points

def cord_to_img(points):
    """
    Transform points from x, y cord axis to image pixel cord (row, col)
    :param points: array of points in format [x, y] of cord axis
    :return: Points in format of (row, col) of image axis
    """
    new_points = np.array([0, IMAGE_SHAPE[1]]) - (points * np.array([-1, 1]))
    # convert col, row to row, col
    new_points = new_points[:, ::-1]
    return new_points

class GraspTriangles:
    """
    Convenience class for loading and operating on sets of Grasp Triangles.
    """

    def __init__(self, grs=None):
        if grs:
            self.grs = grs
        else:
            self.grs = []

    def __getitem__(self, item):
        return self.grs[item]

    def __iter__(self):
        return self.grs.__iter__()

    def __getattr__(self, attr):
        """
        Test if GraspTriangle has the desired attr as a function and call it.
        """
        if hasattr(GraspTriangle, attr) and callable(getattr(GraspTriangle, attr)):
            return lambda *args, **kwargs: list(map(lambda gr: getattr(gr, attr)(*args, **kwargs), self.grs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingTriangles or BoundingTriangle" % attr)
        
    @classmethod
    def load_from_array(cls, arr):
        """
        Load grasp triangles from numpy array.
        :param arr: Nx3x2 array, where each 3x2 array is the 3 corner pixels of a grasp triangle.
        :return: GraspTriangles()
        """
        grs = []
        for i in range(arr.shape[0]):
            grp = arr[i, :, :].squeeze()
            if grp.max() == 0:
                break
            else:
                grs.append(GraspTriangle(grp))
        return cls(grs)
    
    @classmethod
    def load_from_wpi_file(cls, fname):
        """
        Load grasp triangles from a WPI dataset grasp file.
        :param fname: Path to text file.
        :param scale: Scale to apply (e.g. if resizing images)
        :return: GraspTriangle()
        """
        grs = []
        with open(fname) as f:
            while True:
                # Load 3 lines at a time, corners of bounding triangle.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2 = f.readline(), f.readline()
                try:
                    gr = np.array([
                        _gr_text_to_no(p0),
                        _gr_text_to_no(p1),
                        _gr_text_to_no(p2)
                    ])
                    grs.append(GraspTriangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue
        return cls(grs)

    def append(self, gr):
        """
        Add a grasp triangle to this GraspTriangles object
        :param gr: GraspTriangle
        """
        self.grs.append(gr)

    def copy(self):
        """
        :return: A deep copy of this object and all of its GraspTriangles.
        """
        new_grs = GraspTriangles()
        for gr in self.grs:
            new_grs.append(gr.copy())
        return new_grs

    def show(self, ax=None, shape=None):
        """
        Draw all GraspTriangles on a matplotlib plot.
        :param ax: (optional) existing axis
        :param shape: (optional) Plot shape if no existing axis
        """
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            ax.imshow(np.zeros(shape))
            ax.axis([0, shape[1], shape[0], 0])
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def draw(self, shape, position=True, angle=True, width=True, length=True):
        """
        Plot all GraspTriangles as solid triangles in a numpy array, e.g. as network training data.
        :param shape: output shape
        :param position: If True, Q output will be produced
        :param angle: If True, Angle output will be produced
        :param width: If True, Width output will be produced
        :param length: If True, Length output will be produced
        :return: Q, Angle, Width, Length outputs (or None)
        """
        if position:
            pos_out = np.zeros(shape)
        else:
            pos_out = None
        if angle:
            ang_out = np.zeros(shape)
        else:
            ang_out = None
        if width:
            width_out = np.zeros(shape)
        else:
            width_out = None
        if length:
            length_out = np.zeros(shape)
        else:
            length_out = None

        for gr in self.grs:
            rr, cc = gr.disk_coords(shape)
            if position:
                pos_out[rr, cc] = 1.0
            if angle:
                ang_out[rr, cc] = gr.angle
            if width:
                width_out[rr, cc] = gr.width
            if length:
                length_out[rr, cc] = gr.length

        return pos_out, ang_out, width_out, length_out

    def to_array(self, pad_to=0):
        """
        Convert all GraspTriangles to a single array.
        :param pad_to: Length to 0-pad the array along the first dimension
        :return: Nx3x2 numpy array
        """
        a = np.stack([gr.points for gr in self.grs])
        if pad_to:
            if pad_to > len(self.grs):
                a = np.concatenate((a, np.zeros((pad_to - len(self.grs), 3, 2))))
        return a.astype(np.int32)

    @property
    def corner(self):
        """
        Compute mean corner of all GraspTriangles.
        :return: int, mean centre of all GraspTriangles
        """
        points = [gr.corner for gr in self.grs]
        points = np.array(points)
        return np.mean(points, axis=0).astype(np.int32)
    
class GraspTriangle:
    """
    Representation of a grasp in the common "Grasp Triangle" format.
    """

    def __init__(self, points):
        self.points = points # 3x2 numpy array of points in pixel coordinates

    def __str__(self):
        return str(self.points)

    @property
    def angle(self):
        """
        :return: Angle of the grasp to the horizontal. [-pi, pi]
        """
        # Convert pixel coordinates to x, y coordinates
        points = img_to_cord(self.points)
        dx = points[1, 0] - points[0, 0]
        dy = points[1, 1] - points[0, 1]
        return np.arctan2(dy, dx)

    @property
    def as_grasp(self):
        """
        :return: GraspTriangle converted to a Grasp
        """
        return Grasp(self.corner, self.angle, self.length, self.width)

    @property
    def corner(self):
        """
        :return: Triangle main corner point as row, column.
        """
        return self.points[0, :].astype(np.int32)

    @property
    def length(self):
        """
        :return: Triangle length (i.e. along the axis of the grasp)
        """
        points = img_to_cord(self.points)
        dx = points[1, 0] - points[0, 0]
        dy = points[1, 1] - points[0, 1]
        numerator = abs((dx * (points[2, 1] - points[0, 1])) - \
                        (dy * (points[2, 0] - points[0, 0])))
        denominator = np.sqrt(dx ** 2 + dy ** 2)
        return numerator / denominator

    @property
    def width(self):
        """
        :return: Triangle width (length of shortest side)
        """
        points = img_to_cord(self.points)
        dx = points[1, 0] - points[0, 0]
        dy = points[1, 1] - points[0, 1]
        return np.sqrt((dx ** 2) + (dy ** 2))
    
    def disk_coords(self, shape=None):
        """
        :param shape: Output Shape
        :return: Indices of pixels within the grasp corner disk.
        """
        return disk(self.corner, DISK_RADIUS, shape=shape)

    def iou(self, gr, angle_threshold=np.pi / 6):
        """
        Compute IoU with another grasping triangle
        :param gr: GraspingTriangle to compare
        :param angle_threshold: Maximum angle difference between GraspTriangles
        :return: IoU between Grasp Triangles
        """
        if abs(self.angle - gr.angle) > angle_threshold:
            return 0

        rr1, cc1 = self.disk_coords()
        rr2, cc2 = disk(gr.corner, DISK_RADIUS)

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except:
            return 0

        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas == 2)
        return intersection / union

    def copy(self):
        """
        :return: Copy of self.
        """
        return GraspTriangle(self.points.copy())

    def offset(self, offset):
        """
        Offset grasp triangle
        :param offset: array [row, col] distance to offset
        """
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        """
        Rotate grasp triangle
        :param angle: Angle to rotate (in radians)
        :param center: Point to rotate around (e.g. image center)
        """
        points = img_to_cord(self.points)
        c = img_to_cord(np.array(center).reshape((1, 2)))
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        rotated_points = (np.dot(R, (points - c).T)).T + c
        self.points = cord_to_img(rotated_points).astype(np.int32)

    def scale(self, factor):
        """
        :param factor: Scale grasp triangle row and col by factor [0] and factor [1]
        """
        self.points = (self.points * np.array(factor)).astype(np.int32)

    def plot(self, ax, color=None):
        """
        Plot grasping triangle.
        :param ax: Existing matplotlib axis
        :param color: matplotlib color code (optional)
        """
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color)

    def zoom(self, factor, center):
        """
        Zoom grasp triangle by given factor.
        :param factor: Zoom factor
        :param center: Zoom zenter (focus point, e.g. image center)
        """
        points = img_to_cord(self.points)
        c = img_to_cord(np.array(center).reshape((1, 2)))
        T = np.array(
            [
                [1 / factor, 0],
                [0, 1 / factor]
            ]
        )
        zoomed_points = (np.dot(T, (points - c).T)).T + c
        self.points = cord_to_img(zoomed_points).astype(np.int32)

class Grasp:
    """
    A Grasp represented by a corner pixel, rotation angle and triangle width and length.
    """

    def __init__(self, corner, angle, length=60, width=30):
        self.corner = corner # (row, column)
        self.angle = angle  # Positive angle means rotate anti-clockwise from horizontal.
        self.length = length
        self.width = width

    @property
    def as_gt(self):
        """
        Convert to GraspTriangle
        :return: GraspTriangle representation of grasp.
        """
        corner = img_to_cord(np.array(self.corner).reshape((1, 2)))
        points = np.array([
            [corner[0][0], corner[0][1]],
            [corner[0][0] + self.width, corner[0][1]],
            [corner[0][0] + (self.width/2), corner[0][1] - self.length],
        ])

        R = np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)],
            ]
        )
        points_r = ((np.dot(R, (points - corner).T)).T + corner)
        pixel_points = cord_to_img(points_r).astype(np.int32)
        return GraspTriangle(pixel_points)

    def max_iou(self, grs):
        """
        Return maximum IoU between self and a list of GraspTriangles
        :param grs: List of GraspTriangles
        :return: Maximum IoU with any of the GraspTriangles
        """
        self_gr = self.as_gt
        max_iou = 0
        for gr in grs:
            iou = self_gr.iou(gr)
            max_iou = max(max_iou, iou)
        return max_iou

    def plot(self, ax, color=None):
        """
        Plot Grasp
        :param ax: Existing matplotlib axis
        :param color: (optional) color
        """
        self.as_gt.plot(ax, color)


def detect_grasps(q_img, ang_img, width_img=None, length_img=None, no_grasps=1):
    """
    Detect grasps in a network output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)

    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        grasp_angle = ang_img[grasp_point]

        g = Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.width = width_img[grasp_point]
        if length_img is not None:
            g.length = length_img[grasp_point]
        grasps.append(g)

    return grasps
