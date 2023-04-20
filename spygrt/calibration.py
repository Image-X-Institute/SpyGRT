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
# Setting up a logger that will write log messages in the 'stream.log' file.
logger = logging.getLogger(__name__)
fh = logging.FileHandler('stream.log', 'w', 'utf-8')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('- %(module)s - %(levelname)-8s: %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

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

    def __init__(self, stream):
        """

        Args:
            stream: Stream object for the camera to be calibrated.
        """
        # Stream object to query stream setting/parameters. Private as we want to control what method of the stream
        # a Calibrator object can access. Potentially code breaking if
        self._stream = stream
        # Intrinsic matrix of the camera
        # Pose of the camera - modified by the different calibration methods.
        self._pose = o3d.core.Tensor(np.identity(4), dtype=o3d.core.Dtype.Float32, device=DEVICE)
        self.corners = []
        self._mean_corners = None

    @property
    def pose(self):
        """
        Access the 4x4 matrix representing the camera pose (result of the calibration).

        Returns:
            pose (numpy ndarray): a 4x4 matrix representing the input camera to world transform

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

    def write_cal(self, filename=None):
        """
        Write the pose matrix of the camera into a text file.

        Args:
            filename:(string) Filename for the text file in which the calibration is to be saved.
        """
        if filename is None:
            filename = self._stream.serial + "_cal.txt"

        np.savetxt(filename, self.T.numpy())

    def find_corners3d(self, frame, col=DEFAULT_COLUMN, rows=DEFAULT_ROW, draw=False):
        """
        Args:
            frame: (pyrealsense2.frame,pyrealsense2.frame) or (open3d.t.geometry.Image, open3d.t.geometry.Image)
                    The result from the get_frames() method or the frame attribute of a Stream object
            col: (int) Number of column on the calibration board.
            rows: (int) Number of rows on the calibration board.
            draw: (bool) Whether the corners should be drawn on the color image.

        Returns:
            corners3d: (numpy.array) 3D coordinate of all the corners of the calibration board

        Identify the 3D coordinates of the corners of a chessboard in the current frame

        """
        corners = np.asarray([])

        # Obtaining the frame data in numpy format
        if type(frame[0]) == o3d.t.geometry.Image:

            if DEVICE == o3d.core.Device.CUDA:
                color = frame[1].as_tensor().cpu().numpy()
                depth = frame[0].as_tensor().cpu().numpy()
            else:
                color = frame[1].as_tensor().numpy()
                depth = frame[0].as_tensor().numpy()

        else:
            depth = np.asanyarray(frame[0].get_data())*0.001
            color = np.asanyarray(frame[1].get_data())

        ret, corners = cv.findChessboardCorners(color, [col, rows], corners,
                                                cv.CALIB_CB_FAST_CHECK)
        corners3d = []
        for corner in corners:
            pixel = [int(np.rint(corner[0][1])), int(np.rint(corner[0][0]))]
            corners3d.append(rs2.rs2_deproject_pixel_to_point(self._stream.get_rs_intrinsics(), corner[0],
                                                              float(depth[pixel[0], pixel[1]])))

        corners3d = np.asarray(corners3d)

        if draw:
            color = o3d.t.geometry.Image(o3d.core.Tensor(cv.drawChessboardCorners(color, [col, rows], corners, ret),
                                                         device=DEVICE))
        self.corners.append(corners3d)
        return corners3d

    def avg_corners(self, force=False):
        if self._mean_corners is None or force:
            self._mean_corners = np.mean(self.corners, axis=0)

        return self._mean_corners

    def get_corners(self):
        """
        Get the 3D coordinate of coordinates in the world frame of reference.

        Returns: (numpy.array) Array of the corners in the world frame of reference

        """
        corners = np.vstack((self.corners.T, np.ones(len(self.corners)))).T
        corners = self.pose.numpy()@corners.T
        return corners[0:3].T

    def corners_scale(self, size=DEFAULT_SQUARE_SIZE):
        """Class to scale the corners to the real corner size - used to refine calibration."""
        dist = []
        corners = self.avg_corners()
        board = np.reshape(corners, [DEFAULT_ROW, DEFAULT_COLUMN, 3])
        for x in board:
            for i in range(len(x)-1):
                dist.append(np.linalg.norm(x[i]-x[i+1]))

        for x in board.transpose(1, 0, 2):
            for i in range(len(x)-1):
                dist.append(np.linalg.norm(x[i]-x[i+1]))
        scale = size*1/np.mean(dist)
        s_mat = np.identity(4)*scale
        s_mat[3, 3] = 1
        self._pose = o3d.core.Tensor(s_mat, dtype=o3d.core.Dtype.Float32, device=DEVICE)@self.pose

    def align_cameras(self, target):
        """ Find the transform between the two cameras based on the corners of the chessboard.

        If the stream has ended, update the point cloud with the last frame.
        """

        s_corners = self.avg_corners()
        t_corners = target.avg_corners()
        s_corners = o3d.core.Tensor(s_corners, dtype=o3d.core.Dtype.Float32, device=DEVICE)
        t_corners = o3d.core.Tensor(t_corners, dtype=o3d.core.Dtype.Float32, device=DEVICE)
        s_pcd = o3d.t.geometry.PointCloud(s_corners)
        t_pcd = o3d.t.geometry.PointCloud(t_corners)
        corr = np.zeros((len(s_corners), 1), dtype=np.int64)
        corr[:, 0] = np.arange(0, len(s_corners))
        icp = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
        pose = icp.compute_transformation(s_pcd, t_pcd, o3d.core.Tensor(corr, device=DEVICE)).\
            to(o3d.core.Dtype.Float32).to(device=DEVICE)
        self._pose = pose@self.pose

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
        y_axis = np.mean(y_axes, axis=0)*-1
        z_axis = np.cross(x_axis,y_axis)

        # Ensures x,y and z are orthogonal and form a basis to the new coordinate system
        y_axis = np.cross(z_axis, x_axis)

        # Normalization
        x_axis = x_axis/np.linalg.norm(x_axis)
        y_axis = y_axis/np.linalg.norm(y_axis)
        z_axis = z_axis/np.linalg.norm(z_axis)

        # Find the Rotation matrix

        rot_mat = np.array([x_axis, y_axis, z_axis]).T

        # Set the origin to the mean of all corners - Transform to meters.
        origin = np.mean(corners3d, axis=0)

        # Transform rotation and translation matrix to an affine Transformation matrix.
        pose = o3d.core.Tensor(np.linalg.inv(to_affine(rot_mat, origin)), dtype=o3d.core.Dtype.Float32, device=DEVICE)

        self._pose = pose@self.pose
        if cal2 is not None:
            cal2._pose = pose@cal2.pose

# Helper functions


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
    if np.dot(axis,diff) < 0:
        axis = axis * -1

    return axis


class CalibratorOld:
    """Class to contain and perform the calibration of a camera to a predetermined frame of reference or to that of a
    different camera."""

    def __init__(self, cam):
        self.sensor = cam
        self.sensor.start_stream()
        self.depth, self.color = cam.get_frames()
        self.depth = self.depth.filter_bilateral(10, 20 / 65535, 10)
        self.o3d_intrinsics = cam.get_o3d_intrinsics()
        self.T = o3d.core.Tensor(np.identity(4), dtype=o3d.core.Dtype.Float32)
        self.pcd = None
        self.update_pcd()
        try:
            self.corners = self.find_corners3D()
        except TypeError:
            print("Couldn't find corners with default values, please try again with different values")
        self.corners_scale()

    def write_cal(self, filename=None):
        """Write the current calibration to a text file."""
        if filename is None:
            filename = self.sensor.serial + "_cal.txt"

        np.savetxt(filename, self.T.numpy())

    def get_corners(self):
        """Get the 3D coordinate of cordinate in the frame of reference of the current Calibrator object.
        """
        corners = np.vstack((self.corners.T, np.ones(len(self.corners)))).T
        corners = self.T.numpy() @ corners.T
        return corners[0:3].T

    def load_cal(self, filename=None):
        """Load calibration from a text file."""
        if filename is None:
            filename = self.sensor.serial + "_cal.txt"
        T = np.loadtxt(filename)
        self.T = o3d.core.Tensor(T, dtype=o3d.core.Dtype.Float32)

    def refresh(self):
        """Fetch a new frame from the camera.

        Warnings: This function can raise an unchecked exception if stream has ended.
        """
        self.depth, self.color = self.sensor.get_frames()
        self.depth = self.depth.filter_bilateral(10, 20 / 65535, 10)

    def update_pcd(self, limit=np.asarray([1, 1, 1])):
        """Update the Calibrator's pointcloud object to a pointcloud of the current view.

        If the stream has ended, update the pointcloud with the last frame.
        """
        try:
            self.refresh()
        except NoMoreFrames:
            print("Stream has ended, updating pointcloud with the stream's last frame.")
        I = o3d.core.Tensor(np.identity(4), o3d.core.Dtype.Float32)
        rgbd = o3d.t.geometry.RGBDImage(self.color, self.depth, aligned=True)
        pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd,
                                                               o3d.core.Tensor(self.o3d_intrinsics.intrinsic_matrix,
                                                                               dtype=o3d.core.Dtype.Float32),
                                                               I, 1000 / 65535, depth_max=1.2).transform(
            self.T).to_legacy()
        box = o3d.geometry.AxisAlignedBoundingBox(-1 * limit, limit)
        pcd = pcd.crop(box)
        self.pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)

    def find_corners3D(self, col=DEFAULT_COLUMN, rows=DEFAULT_ROW, draw=True):
        """Identify the 3D coordinates of the corners of a chessboard in the current frame"""
        corners = np.asarray([])
        if DEVICE == o3d.core.Device.CUDA:
            color = self.color.as_tensor().cpu().numpy()
            depth = self.depth.as_tensor().cpu().numpy()
        else:
            color = self.color.as_tensor().numpy()
            depth = self.depth.as_tensor().numpy()
        ret, corners = cv.findChessboardCorners(color, [col, rows], corners,
                                                cv.CALIB_CB_FAST_CHECK)
        corners3d = []
        for corner in corners:
            pixel = [int(np.rint(corner[0][1])), int(np.rint(corner[0][0]))]
            corners3d.append(rs2.rs2_deproject_pixel_to_point(self.sensor.intrinsics, corner[0],
                                                              float(depth[pixel[0], pixel[1]])))

        # Rescaling the corners to have coordinate in meters
        corners3d = np.asarray(corners3d) * 65535 / 1000

        if draw:
            self.color = o3d.t.geometry.Image(o3d.core.Tensor(cv.drawChessboardCorners(color,
                                                                                       [col, rows], corners, ret)))

        return corners3d

    def corners_scale(self, size=DEFAULT_SQUARE_SIZE):
        """Class to scale the corners to the real corner size - used to refine calibration."""
        dist = []
        corners = self.get_corners()
        board = np.reshape(corners, [DEFAULT_ROW, DEFAULT_COLUMN, 3])
        for x in board:
            for i in range(len(x) - 1):
                dist.append(np.linalg.norm(x[i] - x[i + 1]))

        for x in board.transpose(1, 0, 2):
            for i in range(len(x) - 1):
                dist.append(np.linalg.norm(x[i] - x[i + 1]))
        scale = size * 1 / np.mean(dist)
        T = np.identity(4) * scale
        T[3, 3] = 1
        self.T = o3d.core.Tensor(T, dtype=o3d.core.Dtype.Float32) @ self.T

    def align_cameras(self, target):
        """ Find the transform between the two cameras based on the corners of the chessboard.

        If the stream has ended, update the pointcloud with the last frame.
        """
        try:
            self.refresh()
        except NoMoreFrames:
            print("Stream has ended, aligning cameras with the stream's last frame.")
        try:
            target.refresh()
        except NoMoreFrames:
            print("Target stream has ended, aligning cameras with the target stream's last frame.")
        s_corners = self.get_corners()
        t_corners = target.get_corners()
        s_corners = o3d.core.Tensor(s_corners, dtype=o3d.core.Dtype.Float32)
        t_corners = o3d.core.Tensor(t_corners, dtype=o3d.core.Dtype.Float32)
        s_pcd = o3d.t.geometry.PointCloud(s_corners).to_legacy()
        t_pcd = o3d.t.geometry.PointCloud(t_corners).to_legacy()
        corr = np.zeros((len(s_corners), 2))
        corr[:, 0] = np.arange(0, len(s_corners))
        corr[:, 1] = np.arange(0, len(t_corners))
        icp = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        T = o3d.core.Tensor(icp.compute_transformation(s_pcd, t_pcd, o3d.utility.Vector2iVector(corr)),
                            dtype=o3d.core.Dtype.Float32)
        self.T = T @ self.T

    def align_to_board(self, cal2=None, col=DEFAULT_COLUMN, rows=DEFAULT_ROW):
        """Find the transform that aligns the sensor's frame of reference to a frame of reference that is defined by a chessboard

            The X axis is align with the rows of the chessboard
            The Y axis is align with the columns of the chessboard
            The Z axis is perpendicular to the chessboard
        """
        corners3d = self.get_corners()

        ###Debug
        # rgbd = o3d.t.geometry.RGBDImage(self.color, self.depth)
        # pcd1 = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd,o3d.core.Tensor(self.o3d_intrinsics.intrinsic_matrix), depth_scale = 1, depth_max = 1500)
        # pcd2 = o3d.t.geometry.PointCloud(o3d.core.Tensor(corners3d))
        ##o3d.visualization.draw_geometries([pcd1.to_legacy(),pcd2.to_legacy()])
        # o3d.t.io.write_point_cloud(str(pixel[0]) + '.ply', pcd2)
        # o3d.io.write_point_cloud(str(pixel[1]) + '.ply', pcd1.to_legacy())

        board = np.reshape(corners3d, [rows, col, 3])
        x_axes = [ms3Dline(line) for line in board]

        # Taking the mean of axis vector of all rows
        x_axis = np.mean(x_axes, axis=0)

        y_axes = [ms3Dline(line) for line in board.transpose(1, 0, 2)]

        # Taking the mean of axis vector of all columns (by design y points in the wrong direction so we correct it by *-1
        y_axis = np.mean(y_axes, axis=0) * -1

        z_axis = np.cross(x_axis, y_axis)

        # Ensures x,y and z are orthogonal and form a basis to the new coordinate system
        y_axis = np.cross(z_axis, x_axis)

        # normalization
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Find the Rotation matrix

        R = np.array([x_axis, y_axis, z_axis]).T

        # Set the origin to the mean of all corners - Transform to meters.
        origin = np.mean(corners3d, axis=0)

        # Transform rotation and translation matrix to an affine Transformation matrix.
        T = o3d.core.Tensor(np.linalg.inv(to_affine(R, origin)), dtype=o3d.core.Dtype.Float32)

        self.T = T @ self.T
        if cal2 is not None:
            cal2.T = T @ cal2.T