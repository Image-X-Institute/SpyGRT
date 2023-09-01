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
        lower = [np.min([p1[0], p2[0]])-0.02, np.min([p1[1], p2[1]])-0.02, mid_z - 0.02]
        upper = [np.max([p1[0], p2[0]])+0.02, np.max([p1[1], p2[1]])+0.02, mid_z + 0.1]

        self._roi = np.asarray([lower, upper], dtype='float32')

    def get_motion(self, source):
        """
        Uses ICP to calculate the motion between the input point cloud and the reference point cloud.

        Args:
            source: Point cloud to be registered.

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
        source = source.crop(box)
        if source.is_empty():
            # WARNING
            return o3d.core.Tensor(np.identity(4), device=DEVICE)
        source.estimate_normals(30, 0.002)
        self._source = source

        result_icp = o3d_reg.multi_scale_icp(self._ref_surface, source, self._icp_config[0], self._icp_config[1],
                                             self._icp_config[2], self._t_guess, self._icp_config[3])


        if hasattr(o3d, 'cuda'):
            self._t_guess = result_icp.transformation.cuda()
        else:
            self._t_guess = result_icp.transformation
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
    def compute_rotation(T):
        """Extract rotations from an affine transformation matrix."""

        if T.is_cuda:
            T = T.cpu()

        R = T.numpy()[0:3].T[0:3].T.copy()
        if not Tracker.isR_matrix(R):
            print("Matrix is not a rotation matrix", sys.stderr)
            sys.exit()

        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            rx = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi
            ry = np.arctan2(-R[2, 0], sy) * 180 / np.pi
            rz = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi

        else: # May need to double-check axes here
            rx = np.arctan2(-R[1, 2], R[1, 1]) * 180 / np.pi
            ry = np.arctan2(-R[2, 0], sy) * 180 / np.pi
            rz = 0

        return [rx, ry, rz]


class TrackerOld:
    """Class to handle surface tracking"""

    def __init__(self, devices, cam_dict=None):
        cam1, cam2 = devices
        if cam_dict is not None:
            if cam1.serial == cam_dict['right']:
                self.r_sensor = cam1
                self.l_sensor = cam2
            elif cam1.serial == cam_dict['left']:
                self.r_sensor = cam2
                self.l_sensor = cam1
            else:
                print("Expected cameras not found, will assign position based on input order from right to left")
                self.r_sensor = cam1
                self.l_sensor = cam2
        else:
            print("WARNING: no input camera directory, camera position will be assigned position based on input order "
                  "from right to left")
            self.r_sensor = cam1
            self.l_sensor = cam2

        self.refresh_frames()
        self.r_o3d_intrinsics = self.r_sensor.get_o3d_intrinsics()
        self.l_o3d_intrinsics = self.l_sensor.get_o3d_intrinsics()

        self.r_T = self.load_cal(CALIBRATION_FOLDER + self.r_sensor.serial + "_cal.txt")
        self.l_T = self.load_cal(CALIBRATION_FOLDER + self.l_sensor.serial + "_cal.txt")

        # This should be replaced by an ROI selection tool.
        self.compute_pcd(u_limit=[0.30, 0.30, 0.30], l_limit=[-0.30, -0.30, 0.01])
        self.ref_pcd = self.pcd

    def refresh_frames(self):
        """Fetch the next frame from each sensor and apply a bilateral filter to the depth data.

        Warnings: This function can raise an unchecked exception if stream has ended.
        """
        self.r_depth, self.r_color = self.r_sensor.get_frames()
        self.l_depth, self.l_color = self.l_sensor.get_frames()

        self.r_depth = self.r_depth.filter_bilateral(10, 20 / 65535, 10).linear_transform(scale=65535 / 1000)
        self.l_depth = self.l_depth.filter_bilateral(10, 20 / 65535, 10).linear_transform(scale=65535 / 1000)
        self.r_frame = o3d.t.geometry.RGBDImage(self.r_color, self.r_depth, aligned=True)
        self.l_frame = o3d.t.geometry.RGBDImage(self.l_color, self.l_depth, aligned=True)

    def compute_pcd(self, limit=None, u_limit=None, l_limit=None, max_depth=1.2):
        """Compute the point cloud captured in the current frame."""

        I = o3d.core.Tensor(np.identity(4), o3d.core.Dtype.Float32)
        if u_limit is not None and l_limit is not None:
            l_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(self.l_frame,
                                                                     o3d.core.Tensor(
                                                                         self.l_o3d_intrinsics.intrinsic_matrix,
                                                                         dtype=o3d.core.Dtype.Float32),
                                                                     I, 1, depth_max=max_depth,
                                                                     with_normals=True).transform(self.l_T).to_legacy()
            r_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(self.r_frame,
                                                                     o3d.core.Tensor(
                                                                         self.r_o3d_intrinsics.intrinsic_matrix,
                                                                         dtype=o3d.core.Dtype.Float32),
                                                                     I, 1, depth_max=max_depth,
                                                                     with_normals=True).transform(self.r_T).to_legacy()
            box = o3d.geometry.AxisAlignedBoundingBox(l_limit, u_limit)
            r_pcd = r_pcd.crop(box)
            l_pcd = l_pcd.crop(box)
            r_pcd = o3d.t.geometry.PointCloud.from_legacy(r_pcd)
            l_pcd = o3d.t.geometry.PointCloud.from_legacy(l_pcd)
            self.pcd = r_pcd + l_pcd
            self.pcd = self.pcd.voxel_down_sample(voxel_size=0.001)

        elif limit is not None:
            l_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(self.l_frame,
                                                                     o3d.core.Tensor(
                                                                         self.l_o3d_intrinsics.intrinsic_matrix,
                                                                         dtype=o3d.core.Dtype.Float32),
                                                                     I, 1, depth_max=max_depth,
                                                                     with_normals=True).transform(self.l_T).to_legacy()
            r_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(self.r_frame,
                                                                     o3d.core.Tensor(
                                                                         self.r_o3d_intrinsics.intrinsic_matrix,
                                                                         dtype=o3d.core.Dtype.Float32),
                                                                     I, 1, depth_max=max_depth,
                                                                     with_normals=True).transform(self.r_T).to_legacy()
            box = o3d.geometry.AxisAlignedBoundingBox(-1 * limit, limit)
            r_pcd = r_pcd.crop(box)
            l_pcd = l_pcd.crop(box)
            r_pcd = o3d.t.geometry.PointCloud.from_legacy(r_pcd)
            l_pcd = o3d.t.geometry.PointCloud.from_legacy(l_pcd)
            self.pcd = r_pcd + l_pcd
            self.pcd = self.pcd.voxel_down_sample(voxel_size=0.001)

        else:
            l_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(self.l_frame,
                                                                     o3d.core.Tensor(
                                                                         self.l_o3d_intrinsics.intrinsic_matrix,
                                                                         dtype=o3d.core.Dtype.Float32),
                                                                     I, 1, depth_max=max_depth,
                                                                     with_normals=True).transform(self.l_T)
            r_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(self.r_frame,
                                                                     o3d.core.Tensor(
                                                                         self.r_o3d_intrinsics.intrinsic_matrix,
                                                                         dtype=o3d.core.Dtype.Float32),
                                                                     I, 1, depth_max=max_depth,
                                                                     with_normals=True).transform(self.r_T)
            self.pcd = r_pcd + l_pcd
            self.pcd = self.pcd.voxel_down_sample(voxel_size=0.001)

    def reinitialize(self, limit=None, u_limit=None, l_limit=None):
        """Reset the reference pointcloud to the current view.

        Warnings: This function can raise an unchecked exception if stream has ended.
        """
        self.refresh_frames()
        if limit is not None:
            self.ref_pcd = self.compute_pcd(limit=limit)
        else:
            self.ref_pcd = self.compute_pcd(u_limit=u_limit, l_limit=l_limit)

    def set_ref_pcd(self, pcd):
        """Set the reference pcd to the input poincloud."""
        self.ref_pcd = pcd

    def ref_motion(self):
        """
        Uses ICP to calculate the motion between the current point cloud and the reference point cloud.
        Input options are to crop the background to increase the accuracy of the registration method.

        NB: This method will not refresh the frame, the caller must call refresh_frames and compute_pcd first
        if the acquisition of a new frame is desired.

        """

        # Initializing Multiscale parameters
        # For downsampling of the pointcloud
        voxel_radius = [0.05, 0.01, 0.005, 0.003, 0.001, 0.0005]
        # Maximum distance between correspondances
        dist = [0.1, 0.05, 0.01, 0.008, 0.005, 0.001]
        # Number of iteration for the RMSE optimization
        max_iter = [50, 30, 14, 7, 3]

        # Initial transformation guess
        T = np.identity(4)

        # Initializing pointclouds
        target = self.ref_pcd
        source = self.pcd

        for scale in range(5):
            iter = max_iter[scale]
            radius = voxel_radius[scale]
            max_distance = dist[scale]

            # Downsample pointclouds
            source_d = source.voxel_down_sample(radius).to_legacy()
            target_d = target.voxel_down_sample(radius).to_legacy()

            # Generalized ICP

            result_icp = o3d.pipelines.registration.registration_generalized_icp(source_d, target_d, max_distance, T,
                                                                                 o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                                                                                 o3d.pipelines.registration.ICPConvergenceCriteria(
                                                                                     relative_fitness=1e-9,
                                                                                     relative_rmse=1e-9,
                                                                                     max_iteration=iter),
                                                                                 )
            T = result_icp.transformation

        return o3d.core.Tensor(T, dtype=o3d.core.Dtype.Float32)

    def ref_track(self, length, filename, limit=None, u_limit=None, l_limit=None, verbose=True):
        """
        Compute the motion between the current view and reference view for a certain duration. The duration is
        given as input in units of # of frames. Input limits parameter can be used to crop the background.
        Verbose input option will enable progress tracking through printing on the console.
        Refer to ref_motion method for tracking details.

        Output is in the form of a text file with the input filename. If the filename doesn't include a complete path,
        it will be saved in the current working directory. Filename should end with the file extension ".txt".
        """

        # Initialize output storage array
        data = []
        for i in range(length):
            try:
                self.refresh_frames()
            except NoMoreFrames:
                print("Stopping loop early as we reached the end of the stream at frame: " + str(i))
                break
            self.compute_pcd(limit=limit, u_limit=u_limit, l_limit=l_limit)
            T = self.ref_motion()
            data.append(np.concatenate((Tracker.compute_translation(T), Tracker.compute_rotation(T))))
            if verbose:
                if i % 5 == 0:
                    print("Finished computing registration for frame:" + str(i))

        np.savetxt(filename, data)

    def load_cal(self, filename):
        T = np.loadtxt(filename)
        return o3d.core.Tensor(T, dtype=o3d.core.Dtype.Float32)

    def new_stream(self, devices, cam_dict=None):
        """Updates the stream object of this tracker with new streamable objects.

        Warnings: This function can raise an unchecked exception if the input is an empty stream.
        """
        cam1, cam2 = devices
        if cam_dict is not None:
            if cam1.serial == cam_dict['right']:
                self.r_sensor = cam1
                self.l_sensor = cam2
            elif cam1.serial == cam_dict['left']:
                self.r_sensor = cam2
                self.l_sensor = cam1
            else:
                print("Expected cameras not found, will assign postion based on input order from right to left")
                self.r_sensor = cam1
                self.l_sensor = cam2
        else:
            print(
                "WARNING: no input camera directory, camera position will be assigned postion based on input order from right to left")
            self.r_sensor = cam1
            self.l_sensor = cam2

        self.refresh_frames()
        self.r_o3d_intrinsics = self.r_sensor.get_o3d_intrinsics()
        self.l_o3d_intrinsics = self.l_sensor.get_o3d_intrinsics()

        self.r_T = self.load_cal(self.r_sensor.serial + "_cal.txt")
        self.l_T = self.load_cal(self.l_sensor.serial + "_cal.txt")

    @staticmethod
    def isR_matrix(R):
        """determine whether a 3x3 matrix is a rotation matrix."""
        if R.is_cuda:
            R = R.cpu()
        return (np.linalg.norm(np.identity(3, dtype=R.dtype) - np.dot(R.T, R)) < 1e-6)

    @staticmethod
    def compute_translation(T):
        """Extract translations from affine tranformation matrix."""
        if T.is_cuda:
            return T.cpu().numpy().T[-1][0:3]
        else:
            return T.numpy().T[-1][0:3]

    @staticmethod
    def compute_rotation(T):
        """Extract rotations from an affine transformation matrix."""
        if T.is_cuda:
            T = T.cpu()
        R = T.numpy()[0:3].T[0:3].T.copy()
        if not Tracker.isR_matrix(R):
            print("Matrix is not a rotation matrix", sys.stderr)
            sys.exit()

        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi
            pitch = np.arctan2(-R[2, 0], sy) * 180 / np.pi
            yaw = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi

        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        return [yaw, pitch, roll]