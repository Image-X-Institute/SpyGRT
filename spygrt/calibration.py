# MIT License

# Copyright (c) 2022 The University of Sydney

# Author: Youssef Ben Bouchta

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2 as cv
import pyrealsense2 as rs2
import open3d as o3d
import numpy as np
import logging

# Modify this if using a calibration board other than the default one available
DEFAULT_ROW = 5
DEFAULT_COLUMN = 8
DEFAULT_SQUARE_SIZE = 0.043

DEVICE = None
if hasattr(o3d, 'cuda'):
    DEVICE = o3d.core.Device('cuda:0')
else:
    DEVICE = o3d.core.Device('cpu:0')


class Calibrator:
    """Class to contain and perform the calibration of a camera to a predetermined frame of reference or to that of a
    different camera."""

    def __init__(self, stream, gt=None):
        """

        Args:
            stream: Stream object for the camera to be calibrated.
            gt: (numpy.ndarray): Array of the ideal position of each corner on the board - first corner is positive
                Y and negative X (IEC coordinate), camera is inf to the calibration board and corner order is row major
        """

        if gt is None:
            rows = DEFAULT_ROW
            col = DEFAULT_COLUMN
            size = DEFAULT_SQUARE_SIZE
            self._gt = np.zeros((rows * col, 3), np.float32)
            self._gt[:, :2] = np.mgrid[0:col, 0:rows].T.reshape(-1, 2) * size
            self._gt[:, 1] = (self._gt[:, 1] - 2 * size)  #* -1
            self._gt[:, 0] = (self._gt[:, 0] - 4 * size) * 1
        else:
            self._gt = gt
        self._switch = False
        # Stream object to query stream setting/parameters. Private as we want to control what method of the stream
        # a Calibrator object can access. Potentially code breaking if used to call get_frames().
        self._stream = stream
        # Intrinsic matrix of the camera
        # Pose of the camera - modified by the different calibration methods.
        self._pose = o3d.core.Tensor(np.identity(4), dtype=o3d.core.Dtype.Float32, device=DEVICE)
        self._epose = None
        self.corners = []
        self._mean_corners = None

    @property
    def pose(self):
        """
        Access the 4x4 matrix representing the camera pose (result of the calibration).

        Returns:
            pose (o3d.core.Tensor): a 4x4 matrix representing the input camera to world transform

        """
        return self._pose

    @pose.setter
    def pose(self, pose):
        """
           Ensures that the pcd attribute is only modified by computing a Point cloud from the current frame.

           Args:
               pose: New pose to replace the existing pose.
        """
        logging.warning('Manually setting the \'pose\' attribute is not allowed, use an alignment method instead')

    @property
    def gt(self):
        """

        Returns:
            gt(numpy.ndarray): a Row*Column by 3 array containing the real position of each corner in row major.
        """
        #temp[:, 0] *= -1
        return self._gt.copy()

    @gt.setter
    def gt(self, gt):
        """

        Args:
            gt: a Row*Column by 3 array containing the real position of each corner in row major.

        """
        self._gt = gt


    def write_cal(self, filename=None):
        """
        Write the pose matrix of the camera into a text file.

        Args:
            filename:(string) Filename for the text file in which the calibration is to be saved.
        """
        if filename is None:
            filename = self._stream.serial + "_cal.txt"

        np.savetxt(filename, self.pose.cpu().numpy())

    def find_corners3d_direct(self, col=DEFAULT_COLUMN, rows=DEFAULT_ROW):
        """
        Use OpenCV's findChessboardCorners function to find the corners in the color frame captured by the camera.
        The 3D position of the corners are then extrapolated from the depth images and added to the 'corners' instance
        variable to be used by the different alignment functions

        Args:
            col: (int) Number of column on the calibration board.
            rows: (int) Number of rows on the calibration board.

        Returns:

        """
        frame = self._stream.frame

        # Obtaining the frame data in numpy format
        if type(frame[0]) == o3d.t.geometry.Image:
            im = frame[1].as_tensor().cpu().numpy()
            color = frame[1].rgb_to_gray().as_tensor().cpu().numpy()
            depth = frame[0].as_tensor().cpu().numpy()

        else:
            depth = np.asanyarray(frame[0].get_data())
            color = np.asanyarray(frame[1].get_data())

        ret, corners = cv_corners(color, alg='SB')
        if not ret:
            ret, corners = cv_corners(color, alg='reg')
        if not ret:
            # Failed to find corners in image, to be consistent with previous implementation
            return None, None, False
        if np.sum(corners[0]) > np.sum(corners[-1]):
            corners = np.flip(np.asarray(corners), axis=0)

        corners3d = []
        for corner in corners:
            [c, r] = [int(np.rint(corner[0][1])), int(np.rint(corner[0][0]))]
            temp = self._stream.intrinsics.inv() @ (float(depth[c, r, 0] * 0.001) *
                                                    o3d.core.Tensor([corner[0, 0], corner[0, 1], 1],
                                                                    dtype=o3d.core.Dtype.Float32,
                                                                    device=DEVICE))
            corners3d.append(temp.cpu().numpy())

        corners3d = np.asarray(corners3d)

        self.corners.append(corners3d)
        return corners, ret

    def find_corners3d(self, col=DEFAULT_COLUMN, rows=DEFAULT_ROW, size=DEFAULT_SQUARE_SIZE, orient=True):
        """
        Use OpenCV's findChessboardCorners function to find the corners in a color frame taken by a camera situated
        directly above the chessboard. This color image is simulated using a raytracing algorithm. The 3D position of
        the corners are then extrapolated from the depth images and added to the 'corners' instance variable to be used
        by the different alignment functions

        The raytracing algorithm requires an initial camera pose estimation - If that isn't available, this algorithm
        will calculate an estimate using the output of the find_corner3d_direct method as input for the align_to_board
        method.

        WARNING: This estimate will be logged in the pose variable of this Calibrator instance.

        Args:
            col: (int) Number of column on the calibration board.
            rows: (int) Number of rows on the calibration board.
            size: (float) Size (in metres) of the length of one square on the calibration board.
            orient: (bool) Activate force orientation with top left corner being the first.

        Returns:
            corners3d: (numpy.array) 3D coordinate of all the corners of the calibration board
        """
        # ISSUE WHERE OPENCV IS INCONSISTENT IN CHOOSING THE FIRST CORNER OF CALIBRATION BOARD. WILL ALWAYS ASSUME
        # WHITE LEFT AND RIGHT CORNERS ARE AT THE TOP THAN GO FROM LEFT TO RIGHT FROM THE PERSPECTIVE OF SOMEONE
        # LOCATED INF OF THE BOARD
        corners = np.asarray([])
        # If there is no initial pose estimate for raytracing.
        if self._epose is None:
            # Get an initial pose estimate based on corners from color image from a single frame.

            corn, ret = self.find_corners3d_direct(col, rows)
            if self.corners is None:
                return None, False



            # WILL NEED A WAY TO SET GT IN CASE RED CORNERS ARE AT THE TOP.
            #gt = self.gt.copy()
            #gt[:, 1] = np.flip(gt[:, 1], axis=0)
            #gt[:, 0] = np.flip(gt[:, 0], axis=0)
            gt = self.gt
            gt[:, 1] = (self._gt[:, 1] - 2 * size) * -1
            temp = self._pose
            self.align_to_board_alt(cal2=None, col=col, rows=rows, size=size, gt=gt)

            self._epose = self._pose
            self._pose = temp
            # Reset the corners variable to remove the corners coordinate from the color frame from future
            # considerations
            self.corners = []
            self._mean_corners = None
            return corn, ret
        else:
            # Compute pcd based on initial guess (see above if statement)
            pcd = self._stream.compute_pcd().transform(self._epose)
            # Position of the viewpoint directly above the board
            ext = o3d.core.Tensor(np.identity(4), dtype=o3d.core.Dtype.Float32, device=DEVICE)
            ext[2, 3] = 0.85  # 85 cm above the board
            #ext[0, 0] = -1
            #ext[1, 1] = -1
            #ext[0, 0] = -1  # To look at corners with top of image towards the head (open cv convention).
            # Create RGB-D image from directly above the board.
            rgbd = pcd.project_to_rgbd_image(1280, 720, self._stream.intrinsics, ext)
            # Interpolation to cover the holes from the change in perspective.
            color = (rgbd.color.dilate(1).as_tensor().cpu().numpy() * 255).astype(np.uint8)
            color = fill(color)
            gray = cv.cvtColor(color, cv.COLOR_RGB2GRAY)
            depth = rgbd.depth.dilate(4).as_tensor().cpu().numpy()

            ### This could be streamlined - could bug if corner finding  algorithm switches in each iteration ###
            ret, corners = cv_corners(gray, alg='SB')
            if not ret:
                ret, corners = cv_corners(gray, alg='reg')
                if not ret:
                    # Failed to find corners in image, to be consistent with previous implementation
                    return None, False

            # Ensures the first corners is always at the top left of the image (neg x and positive y)
            # Can be turned off by setting orient variable to false.
            if np.sum(corners[0]) > np.sum(corners[-1]) and orient:
                corners = np.flip(np.asarray(corners), axis=0)

            corners3d = []
            inv_epose = self._epose.inv()
            for corner in corners:
                [c, r] = [int(np.rint(corner[0][1])), int(np.rint(corner[0][0]))]

                # Temporary variable to hold 3d corners value
                temp = self._stream.intrinsics.inv() @ (float(depth[c, r, 0] * 0.001) *
                                                        o3d.core.Tensor([corner[0, 0], corner[0, 1], 1],
                                                                        dtype=o3d.core.Dtype.Float32,
                                                                        device=DEVICE))

                one = o3d.core.Tensor(1, dtype=o3d.core.Dtype.Float32, device=DEVICE)
                temp = temp.append(one)
                temp = inv_epose@ext.inv()@temp
                temp = temp[0:3]
                corners3d.append(temp.cpu().numpy())

            corners3d = np.asarray(corners3d)
            self.corners.append(corners3d)
            return corners3d, ret

    def avg_corners(self, force=False):
        if self._mean_corners is None or force:
            self._mean_corners = np.mean(self.corners, axis=0)
            self._mean_corners = self._mean_corners.reshape(self._mean_corners.shape[0:2])

        return self._mean_corners

    def get_corners(self, force=False):
        """
        Get the 3D coordinate of corners in the world frame of reference.

        Returns: (numpy.array) Array of the corners in the world frame of reference

        """

        return (self.pose.cpu().numpy()@np.hstack((self.avg_corners(force=force).reshape([40, 3]),
                                                   np.ones(40).reshape([40, 1]))).T)[0:3].T

    # MAY BE DEPRECATED - INDUCES CALIBRATION ERRORS
    def corners_scale(self, size=DEFAULT_SQUARE_SIZE):
        """Class to scale the corners to the real corner size - used to refine calibration."""
        dist = []
        corners = self.avg_corners()
        board = np.reshape(corners, [DEFAULT_ROW, DEFAULT_COLUMN, 3])
        for x in board:
            for i in range(len(x) - 1):
                dist.append(np.linalg.norm(x[i] - x[i + 1]))

        for x in board.transpose(1, 0, 2):
            for i in range(len(x) - 1):
                dist.append(np.linalg.norm(x[i] - x[i + 1]))
        scale = size * 1 / np.mean(dist)
        s_mat = np.identity(4) * scale
        s_mat[3, 3] = 1
        self._pose = o3d.core.Tensor(s_mat, dtype=o3d.core.Dtype.Float32, device=DEVICE) @ self.pose
        #for i, corner in enumerate(corners):
        #    temp = np.append(corner, [1])
        #    self._mean_corners[i] = (s_mat @ temp)[0:3]
        #self._mean_corners.reshape([DEFAULT_ROW*DEFAULT_COLUMN, 3])

    def align_cameras(self, target):
        """ Find the transform between the two cameras based on the corners of the chessboard.
        If the stream has ended, update the point cloud with the last frame.
        """

        s_corners = self.avg_corners(force=True).reshape([40, 3])
        t_corners = target.avg_corners().reshape([40, 3])
        if (self._switch and not target._switch) or (not self._switch and target._switch):
            s_corners[:, 0] *= -1
        s_corners = o3d.core.Tensor(s_corners, dtype=o3d.core.Dtype.Float32, device=DEVICE)
        t_corners = o3d.core.Tensor(t_corners, dtype=o3d.core.Dtype.Float32, device=DEVICE)
        s_pcd = o3d.t.geometry.PointCloud(s_corners)
        t_pcd = o3d.t.geometry.PointCloud(t_corners)
        corr = np.zeros((len(s_corners), 1), dtype=np.int64)
        corr[:, 0] = np.arange(0, len(s_corners))
        icp = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
        pose = icp.compute_transformation(s_pcd, t_pcd, o3d.core.Tensor(corr, device=DEVICE)). \
            to(o3d.core.Dtype.Float32).to(device=DEVICE)
        self._pose = pose @ self.pose

    def align_to_board(self, cal2=None, col=DEFAULT_COLUMN, rows=DEFAULT_ROW):
        """Find the transform that aligns the sensor's frame of reference to a frame of reference that is defined by a chessboard
            The X axis is aligned with the rows of the chessboard
            The Y axis is aligned with the columns of the chessboard
            The Z axis is perpendicular to the chessboard
        """
        corners3d = self.avg_corners()

        board = np.reshape(corners3d, [rows, col, 3])
        x_axes = [ms3dline(line) for line in board]

        # Taking the mean of axis vector of all rows
        x_axis = np.mean(x_axes, axis=0)

        y_axes = [ms3dline(line) for line in board.transpose(1, 0, 2)]

        # Taking the mean axis vector of all columns. The Y-axis needs to be reversed to align to IEC coordinates
        y_axis = np.mean(y_axes, axis=0) * -1
        z_axis = np.cross(x_axis, y_axis)

        # Ensures x,y and z are orthogonal and form a basis to the new coordinate system
        y_axis = np.cross(z_axis, x_axis)

        # Normalization
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Find the Rotation matrix

        rot_mat = np.array([x_axis, y_axis, z_axis]).T

        # Set the origin to the mean of all corners - Transform to metres.
        origin = np.mean(corners3d, axis=0)


        # Transform rotation and translation matrix to an affine Transformation matrix.
        pose = o3d.core.Tensor(np.linalg.inv(to_affine(rot_mat, origin)), dtype=o3d.core.Dtype.Float32, device=DEVICE)

        self._pose = pose @ self.pose
        if cal2 is not None:
            cal2._pose = pose @ cal2.pose

    def align_to_board_alt(self, cal2=None, col=DEFAULT_COLUMN, rows=DEFAULT_ROW, size=DEFAULT_SQUARE_SIZE, gt=None):
        """Find the transform that aligns the sensor's frame of reference to a frame of reference that is defined by a chessboard
            The X axis is aligned with the rows of the chessboard
            The Y axis is aligned with the columns of the chessboard
            The Z axis is perpendicular to the chessboard
        """
        if gt is None:
            gt = self.gt.copy()

        corners3d = self.avg_corners(force=True).reshape([rows*col, 3])

        t_pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(gt, dtype=o3d.core.Dtype.Float32, device=DEVICE))
        s_pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(corners3d, dtype=o3d.core.Dtype.Float32, device=DEVICE))
        s_pcd.transform(self.pose)
        corr = np.zeros((len(gt), 1), dtype=np.int64)
        corr[:, 0] = np.arange(0, len(gt))
        icp = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()

        pose = icp.compute_transformation(s_pcd, t_pcd, o3d.core.Tensor(corr, device=DEVICE)). \
            to(o3d.core.Dtype.Float32).to(device=DEVICE)
        self._pose = pose @ self.pose
        if cal2 is not None:
            cal2._pose = pose @ cal2.pose

    def offset(self, x, y, z):
        """
            Offset the calibration by shifting origin.
        Args:
            x: (double) x offset
            y: (double) y offset
            z: (double) z offset

        """
        offset = o3d.core.Tensor.eye(4, dtype=o3d.core.Dtype.Float32, device=DEVICE)
        offset[0:3, 3] = [x, y, z]
        self._pose = offset@self._pose

    def rotate(self, axis):
        """
            Rotate the calibration axis by 90 degree along the input axis.
        Args:
            axis: Unit vector - must be either the x, y or z axis.

        """

        if np.linalg.norm(axis) != 1:
            print("Input for the rotate function axis: " + str(axis) +" is not a unit vector.")
            return
        if axis[0]**2 == 1:
            rot = o3d.core.Tensor.eye(4, dtype=o3d.core.Dtype.Float32, device=DEVICE)
            rot[1:3, 1] = [0, axis[0]]
            rot[1:3, 2] = axis[0]*[-1*axis[0], 0]
        elif axis[1]**2 == 1:
            rot = o3d.core.Tensor.eye(4, dtype=o3d.core.Dtype.Float32, device=DEVICE)
            rot[0:3, 0] = [0, 0, -1*axis[1]]
            rot[0:3, 2] = [axis[1], 0, 0]
        elif axis[2]**2 == 1:
            rot = o3d.core.Tensor.eye(4, dtype=o3d.core.Dtype.Float32, device=DEVICE)
            rot[0:2, 0] = [0, axis[2]]
            rot[0:2, 1] = [-1*axis[2], 0]
        else:
            print("Input for the rotate function axis: " + str(axis) + " is not aligned with one of the main axis.")
            return

        self._pose = rot@self._pose

# Helper functions
def cv_corners(color, alg='reg', col=DEFAULT_COLUMN, rows=DEFAULT_ROW,):
    """
    Takes an image containing a checkerboard and finds the position of the corners in the image using openCV built-in
    functions.

    Args:
        color(np.ndarray): RGB array containing the image data.
        alg([char]): string identifying the default algorithm to use to get the corners.
        col(int): number of column on the checkerboard
        rows(int): number of rows on the checkerboard..

    Returns:
        ret(bool): Whether the corners could be found
        corners(np.ndarray): Numpy array containing the 2D pixel coordinate of the corners.

    """
    corners = np.asarray([])
    if alg == 'reg':
        ret, corners = cv.findChessboardCorners(color, [col, rows], corners)
        if ret:
            corners = cv.cornerSubPix(color, corners, [11, 11], [-1, -1],
                                      (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1))
    else:
        ret, corners = cv.findChessboardCornersSB(color, [col, rows], corners)
    return ret, corners

def fill(im, ksize=9, copy=False):
    """
      Takes an image resulting from the projection of a point cloud (with holes) and fill the holes with an average
      of the ksize^2 closest pixels that contain colour information.
    Args:
        im: (np.ndarray) image to be filled.
        ksize: (int) size of the kernel (square kernel)
        copy: (bool) Whether to return a copy or not.
    Returns:
        im: (np.ndarray) filled imaged. It is a copy if the copy parameter is true, otherwise it is a modified image
            from  the input image.
    """
    # Threshold for mask
    low = np.asarray([1, 1, 1])
    high = np.asarray([256, 256, 256])

    # copy if we want to return a copy and therefor avoid modifying the input image.
    if copy:
        im = im.copy()

    # Create a mask of all the coloured pixels
    mask = cv.inRange(im, low, high)
    # Change mask value from 255 to 1
    mask[mask == 255] = 1

    # Kernel that will add all the pixel overlapped with the kernel
    sum_ker = np.ones([ksize, ksize])

    # Apply sum_ker to mask, which gives an image where every pixel contains the number of coloured pixel in their
    # neighbourhood (square neighbourhood of size = ksize)
    non_zero_count = cv.filter2D(mask, -1, sum_ker)

    # Broadcast non_zero_count to make it a 3D array (same 2D array repeated 3 times)
    count3d = np.broadcast_to(non_zero_count.reshape(non_zero_count.shape + (1,)), non_zero_count.shape + (3,))

    # Apply sum_ker to the image to obtain an image where the colour of all the pixels in the neighbourhood are added.
    # Converted to float to not truncate RGB values higher than 255.
    sum_im = cv.filter2D(im.astype(np.float32), -1, sum_ker)

    # The condition in brackets below is equivalent to (mask == 0 and non_zero_count !=0). It represents the position
    # of all the black pixels in the original image that have at least one coloured pixel in their neighbourhood.
    # Replace all the pixel that fit the above condition by the average of the coloured pixels in their neighbourhood.
    im[(mask == 0) * (non_zero_count > 0.01)] = (sum_im[(mask == 0) * (non_zero_count > 0.01)] /
                                                 count3d[(mask == 0) * (non_zero_count > 0.01)]).astype(np.uint8)

    return im

def to_affine(r, t):
    """Transforms a 3x3 rotation matrix and a Translation vector to an Affine transformation matrix."""
    return np.vstack((np.vstack((r.T, t)).T, [0, 0, 0, 1]))


def ms3dline(points):
    """
        Args:
            points: numpy array of 3D points.
        Returns:
            axis: numpy array of the 3D direction vector for the line of best fit for the input points.
    Takes in a set of point and find the unit vector in the direction of the best fit line. Positive direction is
    from first point to last point.
    """

    # Translation so all the points are centered at 0
    origin = np.mean(points, axis=0)

    # Vector to help set the direction from first point to last
    diff = points[-1] - points[0]

    # Sets (0,0,0) as the origin
    centered = points - origin

    # Quasi covariance Matrix - different by a factor of sigma squared but irrelevant because I am only after
    # the eigenvectors. Should also should be the hermitian transpose not the transpose, however this will always
    # be a real-valued matrix so they are the same.
    covariance = centered.T @ centered

    # Finds the eigenvalue and eigenvector - eigenvector associated with largest eigenvalue is a vector aligned
    # with line of best fit
    w, v = np.linalg.eigh(covariance)

    # get the eigenvector associated with largest eigenvalue and normalize it
    axis = v.T[-1]
    # Align axis vector with diff (Ensures positive direction is from first point to last point)
    if np.dot(axis, diff) < 0:
        axis = axis * -1

    return axis