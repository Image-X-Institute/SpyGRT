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
from spygrt.stream import NoMoreFrames
import os
import sys

o3d_reg = o3d.t.pipelines.registration
o3d_geo = o3d.t.geometry

# Modify with path to calibration files
CALIBRATION_FOLDER = '../Calibrations/'

DEVICE = None
if hasattr(o3d, 'cuda'):
    DEVICE = o3d.core.Device('cuda:0')
else:
    DEVICE = o3d.core.Device('cpu:0')


class Tracker:
    """Class to handle surface tracking."""

    def __init__(self, ref_surface=None, roi=None):
        # Reference surface for the registration
        if ref_surface is None:
            self._ref_surface = o3d_geo.PointCloud(DEVICE)
        else:
            self._ref_surface = ref_surface

        # ROI boundaries for Registration
        if roi is None:
            # If there is no ROI but a reference surface is
            if ref_surface is not None:
                self._box = None
                self.select_roi()
            else:
                self._roi = np.asarray([[-2., -2., -2.], [2., 2., 2.]], dtype='float32')
        else:
            self._roi = roi

        # Initiating box filter.
        self._box = o3d_geo.AxisAlignedBoundingBox(o3d.core.Tensor(self._roi[0], device=DEVICE),
                                                   o3d.core.Tensor(self._roi[1], device=DEVICE))

        if ref_surface is not None:
            self._ref_surface = self._ref_surface.crop(self._box)
            self._ref_surface.estimate_normals(30, 0.001)

        # Initial transformation guess
        self._t_guess = o3d.core.Tensor(np.identity(4), device=DEVICE)

        # FOR DEBUG AND TESTING OF get_motion.
        self._source = None

        self._icp_config = None

    @property
    def t_guess(self):
        """

        Returns:

        """
        return self._t_guess

    @t_guess.setter
    def t_guess(self, t_guess):
        """

        Args:
            t_guess:

        Returns:

        """
        if type(t_guess) == np.ndarray:
            self._t_guess = o3d.core.Tensor(t_guess, device=DEVICE)
        else:
            self._t_guess = t_guess

    # Might want to create some checks in setters that ensure only valid values are given(i.e no empty point cloud etc.)

    @property
    def ref_surface(self):
        return self._ref_surface

    @ref_surface.setter
    def ref_surface(self, ref_surface):
        self._ref_surface = ref_surface.crop(self._box)
        self._ref_surface.estimate_normals(30, 0.002)

    @property
    def roi(self):
        return self._roi

    @roi.setter
    def roi(self, roi):
        self._roi = np.asarray(roi, dtype='float32')
        self._box.set_min_bound = o3d.core.Tensor(self._roi[0], device=DEVICE)
        self._box.set_max_bound = o3d.core.Tensor(self.roi[1], device=DEVICE)

    def select_roi(self):
        """
        Allow for manual selection of a region of interest to track on the reference surface.

        Returns:
            roi: min and max coordinate of a box containing the ROI.
        """
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(self._ref_surface.to_legacy())
        vis.run()
        vis.destroy_window()
        idx = vis.get_picked_points()
        p1 = self._ref_surface.point["positions"][idx[0]].cpu().numpy()
        p2 = self._ref_surface.point["positions"][idx[1]].cpu().numpy()

        mid_z = np.mean([p1[2], p2[2]])
        lower = [np.min([p1[0], p2[0]]) - 0.02, np.min([p1[1], p2[1]]) - 0.02, mid_z - 0.02]
        upper = [np.max([p1[0], p2[0]]) + 0.02, np.max([p1[1], p2[1]]) + 0.02, mid_z + 0.1]

        self._roi = np.asarray([lower, upper], dtype='float32')

    def get_motion(self, source, robust=True):
        """
        Uses ICP to calculate the motion between the input point cloud and the reference point cloud.

        Args:
            source(o3d.t.geometry.PointCloud): Point cloud to be registered.
            robust(bool): Logical value that determines whether the ICP is tried with an expended ROI when there is
            the surface isn't found inside the ROI.

        Returns:
            motion: 4x4 transformation matrix representing the change in 6DoF motion
        """
        if self._icp_config is None:

            # Initializing Multiscale parameters

            # For down sampling of the point-cloud
            voxel_radius = o3d.utility.DoubleVector([0.05, 0.01, 0.005, 0.003, 0.001])
            # Maximum distance between correspondences
            dist = o3d.utility.DoubleVector([0.1, 0.05, 0.01, 0.008, 0.005])
            # Number of iteration for the RMSE optimization
            max_iter = o3d.utility.IntVector([50, 30, 14, 7, 3])

            # List of convergence criterion for multiscale ICP
            criteria = []
            for it in max_iter:
                criteria.append(o3d_reg.ICPConvergenceCriteria(relative_fitness=1e-9,
                                                               relative_rmse=1e-9,
                                                               max_iteration=it))

            # Point further than 1cm apart are considered non-corresponding.
            loss_function = o3d_reg.robust_kernel.RobustKernel(o3d_reg.robust_kernel.RobustKernelMethod.GeneralizedLoss,
                                                               0.003)
            # ICP specific method (Point to plane + loss function)
            method = o3d_reg.TransformationEstimationPointToPlane(loss_function)

            self._icp_config = [voxel_radius, criteria, dist, method]

        # Cropping the input point cloud to obtain only the ROI
        box = o3d_geo.OrientedBoundingBox.create_from_axis_aligned_bounding_box(self._box)
        # Commented out as transform is not yet implemented.
        box.transform(self._t_guess)
        new_source = source.crop(box)
        if new_source.is_empty() or new_source.point['positions'].shape[0] < self._ref_surface.point['positions']. \
                shape[0] * 0.1:
            # Warning
            if robust:
                box.scale(2)
                new_source = source.crop(box)
                if new_source.is_empty() or new_source.point['positions'].shape[0] < self. \
                        _ref_surface.point['positions'].shape[0] * 0.1:
                    # WARNING
                    self._t_guess = o3d.core.Tensor(np.identity(4), device=DEVICE)
                    return self._t_guess
            else:
                self._t_guess = o3d.core.Tensor(np.identity(4), device=DEVICE)
                return self._t_guess

        new_source.estimate_normals(30, 0.002)
        self._source = new_source

        result_icp = o3d_reg.multi_scale_icp(self._ref_surface, new_source, self._icp_config[0], self._icp_config[1],
                                             self._icp_config[2], self._t_guess, self._icp_config[3])

        if hasattr(o3d, 'cuda'):
            self._t_guess = result_icp.transformation.cuda()
        else:
            self._t_guess = result_icp.transformation
        return self._t_guess

    def get_motion_alt(self, source, robust=True):
        """
        Uses ICP to calculate the motion between the input point cloud and the reference point cloud.

        Args:
            source(o3d.t.geometry.PointCloud): Point cloud to be registered.
            robust(bool): Logical value that determines whether the ICP is tried with an expended ROI when there is
            the surface isn't found inside the ROI.

        Returns:
            motion: 4x4 transformation matrix representing the change in 6DoF motion
        """
        if self._icp_config is None:

            # Initializing Multiscale parameters

            # For down sampling of the point-cloud
            voxel_radius = o3d.utility.DoubleVector([0.05, 0.01, 0.005, 0.003, 0.001])
            # Maximum distance between correspondences
            dist = o3d.utility.DoubleVector([0.1, 0.05, 0.01, 0.008, 0.005])
            # Number of iteration for the RMSE optimization
            max_iter = o3d.utility.IntVector([50, 30, 14, 7, 3])

            # List of convergence criterion for multiscale ICP
            criteria = []
            for it in max_iter:
                criteria.append(o3d_reg.ICPConvergenceCriteria(relative_fitness=1e-9,
                                                               relative_rmse=1e-9,
                                                               max_iteration=it))

            # Point further than 1cm apart are considered non-corresponding.
            loss_function = o3d_reg.robust_kernel.RobustKernel(o3d_reg.robust_kernel.RobustKernelMethod.GeneralizedLoss,
                                                               0.003)
            # ICP specific method (Point to plane + loss function)
            method = o3d_reg.TransformationEstimationPointToPlane(loss_function)

            self._icp_config = [voxel_radius, criteria, dist, method]

        # Cropping the input point cloud to obtain only the ROI
        box = o3d_geo.OrientedBoundingBox.create_from_axis_aligned_bounding_box(self._box)
        # Commented out as transform is not yet implemented.
        box.transform(self._t_guess)
        new_source = source.crop(box)
        if new_source.is_empty() or new_source.point['positions'].shape[0] < self._ref_surface.point['positions']. \
                shape[0] * 0.1:
            # Warning
            if robust:
                box.scale(2)
                new_source = source.crop(box)
                if new_source.is_empty() or new_source.point['positions'].shape[0] < self. \
                        _ref_surface.point['positions'].shape[0] * 0.1:
                    # WARNING
                    self._t_guess = o3d.core.Tensor(np.identity(4), device=DEVICE)
                    return self._t_guess
            else:
                self._t_guess = o3d.core.Tensor(np.identity(4), device=DEVICE)
                return self._t_guess

        #new_source.estimate_normals(30, 0.002)
        self._source = new_source

        result_icp = o3d_reg.multi_scale_icp(new_source, self._ref_surface, self._icp_config[0], self._icp_config[1],
                                             self._icp_config[2], self._t_guess, self._icp_config[3])

        if hasattr(o3d, 'cuda'):
            self._t_guess = result_icp.transformation.cuda()
        else:
            self._t_guess = result_icp.transformation
        self._t_guess = self._t_guess.inv()
        return self._t_guess

    @staticmethod
    def isR_matrix(R):
        """determine whether a 3x3 matrix is a rotation matrix."""
        return np.linalg.norm(np.identity(3, dtype=R.dtype) - np.dot(R.T, R)) < 1e-5

    @staticmethod
    def compute_translation(T):
        """Extract translations from affine tranformation matrix."""
        if T.is_cuda:
            return T.cpu().numpy().T[-1][0:3]
        else:
            return T.numpy().T[-1][0:3]

    @staticmethod
    def compute_rotation(T, conv='XYZ'):
        """Extract rotations from an affine transformation matrix."""

        if T.is_cuda:
            T = T.cpu()

        R = T.numpy()[0:3].T[0:3].T.copy()
        if not Tracker.isR_matrix(R):
            print("Matrix is not a rotation matrix", sys.stderr)
            sys.exit()

        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if conv == 'XYZ':
            sy = np.sqrt(R[0, 0] * R[0, 0] + R[0, 1] * R[0, 1])
            singular = sy < 1e-6
            if not singular:
                rx = np.arctan2(-R[1, 2], R[2, 2]) * 180 / np.pi
                ry = np.arctan2(R[0, 2], sy) * 180 / np.pi
                rz = np.arctan2(-R[0, 1], R[0, 0]) * 180 / np.pi

            else:  # May need to double-check axes here
                rx = np.arctan2(-R[1, 2], R[1, 1]) * 180 / np.pi
                ry = np.arctan2(-R[2, 0], sy) * 180 / np.pi
                rz = 0
        elif conv == 'ZYX':
            sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6
            if not singular:
                rx = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi
                ry = np.arctan2(-R[2, 0], sy) * 180 / np.pi
                rz = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi

            else:  # May need to double-check axes here
                rx = np.arctan2(-R[1, 2], R[1, 1]) * 180 / np.pi
                ry = np.arctan2(-R[2, 0], sy) * 180 / np.pi
                rz = 0

        return [rx, ry, rz]