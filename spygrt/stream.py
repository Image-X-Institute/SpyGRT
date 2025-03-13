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

import os
import threading
import logging
import pyrealsense2 as rs2
import open3d as o3d
import numpy as np
import time
import datetime
from abc import ABC, abstractmethod


DEVICE = None
# ALLOW_GPU can be changed to read from a setting file. If true, GPU will be used if possible.
ALLOW_GPU = os.environ.get('_RTM_GUI_ALLOWGPU')
if ALLOW_GPU is None:
    ALLOW_GPU = "1"
if hasattr(o3d, 'cuda') and ALLOW_GPU == "1":
    DEVICE = o3d.core.Device('cuda:0')
else:
    DEVICE = o3d.core.Device('cpu:0')


class NoMoreFrames(Exception):
    pass


class DualStreamError(Exception):
    pass


class Stream(ABC):
    """Abstract Parent Class to handle the interactions with the RealSense API"""

    @abstractmethod
    def __init__(self, stream_source):
        """Initialise stream and define stream settings."""

    @abstractmethod
    def get_frames(self):
        pass

    @property
    def frame(self):
        pass

    @frame.setter
    def frame(self, new_frame):
        """
        Ensures that the frame attribute is only modified by fetching a new frame from the camera.

        Args:
            new_frame: New frame to replace existing frame.
        """
        logging.warning('Manually setting the \'frame\' attribute is not allowed, use \'get_frame\' instead')

    @property
    @abstractmethod
    def pose(self):
        pass

    @property
    @abstractmethod
    def pcd(self):
        pass

    @pcd.setter
    def pcd(self, pcd):
        """
        Ensures that the pcd attribute is only modified by computing a Point cloud from the current frame.

        Args:
            pcd: New point cloud to replace the existing point cloud.
        """
        logging.warning('Manually setting the \'pcd\' attribute is not allowed, use \'compute_pcd\' instead')

    @abstractmethod
    def compute_pcd(self, new_frame=False):
        pass

    @abstractmethod
    def load_calibration(self, filename=None):
        pass

    @abstractmethod
    def end_stream(self):
        pass

    @abstractmethod
    def start_stream(self):
        pass


class Camera(Stream):
    """Class to handle the interaction with the Realsense API for real-time frame capture"""

    def __init__(self, device, **kwargs):
        """
        Initialise the camera settings and start the stream.
        Args:
            device(rs2.device): Realsense device object pointing to a camera(rs2.device).
            cal_file(string): Filepath to the calibration file
        """
        # logger initialisation
        self.log = logging.getLogger(__name__)

        # Flag to indicate stream is active and warmed up or closed down properly
        self._streamingFlag = threading.Event()

        # Initialising frame attribute.
        self._frame = None
        self._timestamp = None
        self._pose = None
        self._device = device

        # Setting default initialisation option
        init_options = {
            'cal_file': None,
            'emitter_enabled': True,
            'laser_power': 'max',
            'max_distance': 200,
            'min_distance': 0,
            'temporal_smooth_alpha': 1,
            'temporal_smooth_delta': 100,
            'temporal_persistency_idx': 8,
            'threshold_max': 1.3,
            'threshold_min': 0.1,
            'depth_image_width': 1280,
            'depth_image_height': 720,
            'depth_image_format': rs2.format.z16,
            'depth_image_fps': 30,
            'color_image_width': 1280,
            'color_image_height': 720,
            'color_image_format': rs2.format.rgb8,
            'color_image_fps': 30,
            'inter_cam_sync_mode': 0
        }
        for name, value in kwargs.items():
            if name in init_options:
                init_options[name] = value
        if init_options['cal_file'] is not None:
            self.load_calibration(kwargs['cal_file'])


        # Realsense point cloud object useful for the compute_pcd method. For internal class use only.
        self._rs_pc = rs2.pointcloud()

        # A point cloud object representing the latest frame. Stored in a Open3D Tensor PointCloud format.
        self._pcd = o3d.t.geometry.PointCloud(device=DEVICE)


        # Initialise serial number early as it is needed for Stream configuration
        self.serial = device.get_info(rs2.camera_info.serial_number)

        # Indicate whether the camera has warmed up. If false, need to leave the camera running
        # for 10-15frames before any get_frames can be used.
        self.warmed = False

        # pipeline that controls the acquisition of frames.
        self._pipe = rs2.pipeline()

        # Initialise config object
        self._cfg = rs2.config()

        # Configure the camera with the default setting.
        self._cfg.enable_device(self.serial)

        # Configure stream format (resolution, frame rate, etc.)
        # NEED TO TEST TO ENSURE L515 COMPATIBILITY
        self._cfg.enable_stream(rs2.stream.color, 1280, 720, rs2.format.rgb8, 30)
        self._cfg.enable_stream(rs2.stream.depth, 1280, 720, rs2.format.z16, 30)

        # Object that align depth and color.
        self.__align_to_color = rs2.align(rs2.stream.color)

        # Model of the camera that captured this stream.
        self.model = device.get_info(rs2.camera_info.name)


        # Depth sensor object to access the depth scale
        depth_sensor = device.first_depth_sensor()

        # Depth Scale, indicating to convert the data to realworld distances
        self._depth_scale = depth_sensor.get_depth_scale()

        # Pinhole camera intrinsics for depth camera
        self._intrinsics = depth_sensor.get_stream_profiles()[0].as_video_stream_profile().intrinsics

        if depth_sensor.supports(rs2.option.emitter_enabled):
            depth_sensor.set_option(rs2.option.emitter_enabled, init_options['emitter_enabled'])

        if depth_sensor.supports(rs2.option.laser_power):
            if init_options['laser_power'] == 'max':
                laser = depth_sensor.get_option_range(rs2.option.laser_power)
                depth_sensor.set_option(rs2.option.laser_power, laser.max)
            elif init_options['laser_power'] == 'min':
                laser = depth_sensor.get_option_range(rs2.option.laser_power)
                depth_sensor.set_option(rs2.option.laser_power, laser.min)
            else:
                depth_sensor.set_option(rs2.option.laser_power, init_options['laser_power'])

        if depth_sensor.supports(rs2.option.max_distance):
            depth_sensor.set_option(rs2.option.max_distance, init_options['max_distance'])

        if depth_sensor.supports(rs2.option.min_distance):
            depth_sensor.set_option(rs2.option.min_distance, init_options['min_distance'])

        if depth_sensor.supports(rs2.option.inter_cam_sync_mode):
            depth_sensor.set_option(rs2.option.inter_cam_sync_mode, init_options['inter_cam_sync_mode'])

        self._threshold_filter = rs2.threshold_filter(init_options['threshold_min'], init_options['threshold_max'])

        ############################################################
        # TESTING 123
        self._temporal_filter = rs2.temporal_filter(init_options['temporal_smooth_alpha'],
                                                    init_options['temporal_smooth_delta'],
                                                    init_options['temporal_persistency_idx'])

        self._filters = {
            "threshold": self._threshold_filter,
            "temporal": self._temporal_filter
        }

        #########################################################

    @property
    def isStreaming(self):
        return self._streamingFlag.is_set()

    @property
    def frame(self):
        """
        Access the last frame.

        Returns:
            frame: A tuple containing the last frame ordered (depth, colour)
        """
        if self._frame is None:
            self._frame = self.get_frames()
        return self._frame

    @property
    def pose(self):
        """
        Access the 4x4 matrix representing the camera pose.

        Returns:
            pose(o3d.core.Tensor): a 4x4 matrix representing the camera to world transform
        """

        if self._pose is None:
            try:  # Try because we're calling load_calibration without a filename and using the default filename and
                # location
                self.load_calibration()
            except OSError as e:
                logging.error("Could not find the default calibration file for camera with serial number: "
                              + self.serial)
                return o3d.core.Tensor(np.identity(4), dtype=o3d.core.Dtype.Float32, device=DEVICE)
        return self._pose

    @property
    def pcd(self):
        """
        Access the point cloud form of the current frame.

        Returns:
            pcd(o3d.t.geometry.PointCloud): A tensor-based Open3D point cloud object.
        """
        if self._pcd.is_empty():
            self.compute_pcd()
        return self._pcd

    @pcd.setter
    def pcd(self, pcd):
        """
        Ensures pcd is read-only

        Args:
            pcd: New point cloud
        """
        logging.warning("tried to manually set a new point cloud")
        pass

    @property
    def intrinsics(self):
        """
        Access the camera's pinhole camera intrinsics.

        Returns:
            intrinsics(o3d.t.Tensor): 3x3 pinhole camera intrinsics matrix
        """
        return o3d.core.Tensor([[self._intrinsics.fx, 0, self._intrinsics.ppx],
                                [0, self._intrinsics.fy, self._intrinsics.ppy], [0, 0, 1]],
                               dtype=o3d.core.Dtype.Float32, device=DEVICE)

    @property
    def depth_scale(self):
        """
        Conversion factor to go from raw depth data to depth in meters. It is = to 0.001 by default for Realsense
        cameras. Warning: Open3D's depth_scale is = 1/depth_scale.

        Returns:
            depth_scale (float): scaling factor to get depth from Z16 to depth in meters.
        """
        return self._depth_scale

    @property
    def timestamp(self):
        """
        Access the timestamp of the latest frame

        Returns:
            timestamp: Timestamp of the most recent frame (ms), averaged of the two streams' timestamp.
        """

        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp):
        """
        Ensure timestamp is read-only.

        Args:
            timestamp: New timestamp
        """
        logging.warning("Tried to manually change the timestamp")
        pass

    @property
    def sensor_temp(self):
        """
        ASIC depth sensor temperature

        Returns:
            sensor_temp(float): sensor temperature
        """
        return self._pipe.get_active_profile().get_device().\
            first_depth_sensor().get_option(rs2.option.projector_temperature)

    @sensor_temp.setter
    def sensor_temp(self, temp):
        """
        Ensure temperature is read-only.

        Args:
            temp: New temperature value
        """
        logging.warning("Tried to manually change the sensor temperature")
        pass

    @property
    def projector_temp(self):
        """
        Projector depth sensor temperature

        Returns:
            projector_temp(float): projector temperature
        """
        return self._pipe.get_active_profile().get_device().\
            first_depth_sensor().get_option(rs2.option.projector_temperature)

    @projector_temp.setter
    def projector_temp(self, temp):
        """
        Ensure temperature is read-only.

        Args:
            temp: New temperature value
        """

        logging.warning("Tried to manually change the projector temperature")
        pass

    def waitStreaming(self, timeout=3):
        self._streamingFlag.wait(timeout=timeout)

    def compute_pcd(self, new_frame=False):
        """
         Computes a Point Cloud from the newest acquired frame.
         Args:
             new_frame (bool): to determine whether a new frame should be acquired first.
         Returns:
             pcd (Open3d.t.Geometry.PointCloud): The point cloud representation of the newest acquired frame in an
                Open3D Tensor PointCloud format.
         """

        if new_frame:
            self.get_frames('o3d')

        if type(self.frame[0]) == rs2.depth_frame or type(self.frame[0]) == rs2.frame:
            self._rs_pc.map_to(self.frame[1])
            # Get the raw point cloud data
            dat = self._rs_pc.calculate(self.frame[0])
            # Changes how the data is "viewed" without making changes to the memory. Results in a list of 3D coordinates
            depth_dat = np.asanyarray(dat.get_vertices()).view(np.float32).reshape([-1, 3])
            # Reformat the colour data into a list of RGB values
            col_dat = np.asanyarray(self.frame[1].get_data()).reshape([-1, 3])
            col_t = o3d.core.Tensor(col_dat, device=DEVICE)
            depth_t = o3d.core.Tensor(depth_dat, device=DEVICE)
            self._pcd.point["positions"] = depth_t
            self._pcd.point["colors"] = col_t
        else:
            rgbdim = o3d.t.geometry.RGBDImage(self.frame[1], self.frame[0])
            if self._pose is None:
                self._pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbdim, self.intrinsics, depth_scale=1000,
                                                                             depth_max=1.5)
            else:
                self._pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbdim, self.intrinsics,
                                                                             extrinsics=self.pose.inv(),
                                                                             depth_scale=1000,
                                                                             depth_max=1.5)

        return self._pcd

    def load_calibration(self, filename=None):
        """
        Initiates the pose attribute from a calibration file.
        Args:
            filename: String containing the filename of the calibration file.
        Returns:
            pose (numpy ndarray): a 4x4 matrix representing the camera to world transform
        """
        if filename is None:
            filename = self.serial + "_cal.txt"
        pose = np.loadtxt(filename)
        if not pose.shape == (4, 4):
            self.log.error(f"Error loading calibration: {calFile}. Does not conform to 4x4 matrix")
            return
        self._pose = o3d.core.Tensor(pose, device=DEVICE)

        return self._pose

    ############################################################
    # TESTING 123
    def _rs_filter(self, rs_depth, filters):
        """

        Args:
            rs_depth(rs2.depth_frame): Realsense depth frame to be filtered.
            filters(list(str)): List of filters to be applied.

        Returns:
            f_depth(rs2.depth_frame): Filtered Realsense frame

        """
        if filters is None:
            logging.warning("Tried to call a filter without specifying a filter")
            return
        f_depth = rs_depth
        for f in filters:
            f_depth = self._filters[f].process(f_depth)
        return f_depth

    # Encoding will be deprecated later, for now used for backward compatibility
    def get_frames(self, encoding='o3d', filtering=True, filters=["threshold"], timeout=5000):
        """
        Fetch an aligned depth and color frame.
        Args:
            encoding(List[char]): a flag that determines the type of the returned frame.
            filtering(bool): Filters are turned on if set to True
            filters(list(str)): Name of the filter to be applied.
        Returns:
            frame: a tuple with a depth and colour frame ordered (depth, colour). The type of the returned
        frames depend on the input value for encoding.
        """
        if not self.warmed:
            self.warmup()
        frames = self._pipe.wait_for_frames(timeout_ms=timeout)

        aligned_frames = self.__align_to_color.process(frames)

        rs_depth = aligned_frames.get_depth_frame()
        rs_color = aligned_frames.get_color_frame()
        if filtering:
            rs_depth = self._rs_filter(rs_depth, filters)
        self._timestamp = rs_depth.timestamp
        if encoding == 'o3d':
            np_depth = np.asanyarray(rs_depth.get_data())
            np_color = np.asanyarray(rs_color.get_data())
            depth = o3d.t.geometry.Image(o3d.core.Tensor(np_depth, device=DEVICE))
            color = o3d.t.geometry.Image(o3d.core.Tensor(np_color, device=DEVICE))
            self._frame = (depth, color)

        else:
            self._frame = (rs_depth, rs_color)

        # Reset the point cloud as it no longer represents the current frame. This will force calling of
        # compute_pcd method if the pcd attribute is accessed prior to a call to compute_pcd.
        self._pcd = o3d.t.geometry.PointCloud(device=DEVICE)
        return self._frame

    def warmup(self):
        """"Warmup the camera before frame acquisition"""
        for i in range(50):
            try:
                self._pipe.wait_for_frames()
            except Exception as e:
                raise Exception(f"{e}. Please check camera is not connected to another application.")

        self.log.info("Warmed up")
        self.warmed = True
        self._streamingFlag.set()

    def o3d_intrinsics(self):
        """
        Transform the camera intrinsics to an open_3D format.
         Returns:
            intrinsics(o3d.camera.PinholeCameraIntrinsic): Return the intrinsics of the camera in an open3D format.
        """
        return o3d.camera.PinholeCameraIntrinsic(self._intrinsics.width, self._intrinsics.height, self._intrinsics.fx,
                                                 self._intrinsics.fy, self._intrinsics.ppx, self._intrinsics.ppy)

    def get_rs_intrinsics(self):
        """
        Transform the camera intrinsics to an open_3D format.
        Returns:
            intrinsics: Return the intrinsics of the camera in a realsense format.
        """

        return self._intrinsics

    def end_stream(self):
        """End the stream."""

        if not self.isStreaming:
            # not streaming
            return
        try:
            self.log.info("Stopping RS2 pipeline")
            device = self._pipe.get_active_profile().get_device()
            self._pipe.stop()
            self.log.info(f"Stopped: {device}")

        finally:
            # clear state vars
            self.warmed = False
            self._streamingFlag.clear()

    def start_stream(self, rec=False, bag_file=None, warm_up=True):
        """
        Starts or restarts the stream.
        Args:
            rec: Bool to indicate whether a recording should be created
            bag_file: filepath to the desired location for the recorded file.
        """
        if self.isStreaming:
            # already running
            self.log.warning("Aready streaming")
            return

        self.log.warning("Starting RS2 pipeline")

        self._warmed = False

        if rec:
            if bag_file is None:
                bag_file = self.serial + '_' + time.localtime() + '.bag'
            else:
                try:
                    if bag_file[-4:] != '.bag':
                        bag_file = bag_file + '.bag'
                except TypeError:
                    self.log.warning('filename of bag file is not a string, using default filename instead')
                    bag_file = self.serial + time.localtime() + '.bag'
            self._cfg.enable_record_to_file(bag_file)

        self._pipe.start(self._cfg)

        if warm_up:
            self.warmup()


class Recording(Stream):
    """Class to handle the interaction with the Realsense API for realsense bag files."""

    def __init__(self, device, filename=None):
        """
        Args:
            device(str): path a bag file containing a recording.
            filename(str): path to a calibration file
        """
        # logger
        self.log = logging.getLogger(__name__)

        # flag to indicate stream is active and warmed up or closed down properly
        self._streamingFlag = threading.Event()

        self._frame = None

        if filename is not None:
            self.load_calibration(filename)
        else:
            self._pose = None

        self._timestamp = None

        # Realsense point cloud object useful for the compute_pcd method. For internal class use only.
        self._rs_pc = rs2.pointcloud()

        # Realsense threshold filter object for processing frames.
        self._threshold_filter = rs2.threshold_filter(0.1, 1.3)

        # Realsense spatial filter that lower noise and preserve edges.
        self._spatial_filter = rs2.spatial_filter()

        ############################################################
        # TESTING 123
        self._temporal_filter = rs2.temporal_filter(1, 100, 8)

        self._filters = {
            "threshold": self._threshold_filter,
            "temporal": self._temporal_filter,
            "spatial": self._spatial_filter
        }

        # A point cloud object representing the latest frame. Stored in a Open3D Tensor PointCloud format.
        self._pcd = o3d.t.geometry.PointCloud(device=DEVICE)

        # Initialise config object
        self._cfg = rs2.config()

        # Configure the camera with the default setting.
        self._cfg.enable_device_from_file(device, False)

        ctx = rs2.context()

        # Realsense device object to access device metadata.
        self._device = ctx.load_device(device)

        # pipeline that controls the acquisition of frames.
        self._pipe = None

        # Object that align depth and color.
        self._align_to_color = rs2.align(rs2.stream.color)

        # Serial number of the camera that captured this stream.
        self.serial = self._device.get_info(rs2.camera_info.serial_number)

        # Model of the camera that captured this stream.
        self.model = self._device.get_info(rs2.camera_info.name)

        # Access unit in which the depth is saved
        depth_sensor = self._device.first_depth_sensor()

        self._depth_scale = depth_sensor.get_depth_scale()

        # Pinhole camera intrinsics for depth camera
        self._intrinsics = depth_sensor.get_stream_profiles()[0].as_video_stream_profile().intrinsics

        self._playback = None

    @property
    def isStreaming(self):
        return self._streamingFlag.is_set()

    @property
    def frame(self):
        """
        Access the last frame.

        Returns:
            frame: A tuple containing the last frame ordered (depth, colour)
        """
        if self._frame is None:
            self._frame = self.get_frames()
        return self._frame

    @property
    def pose(self):
        """
        Access the 4x4 matrix representing the camera pose.

        Returns:
            pose (o3d.core.Tensor): a 4x4 matrix representing the camera to world transform
        """

        if self._pose is None:
            try:  # Try because we're calling load_calibration without a filename and using the default filename and
                # location
                self.load_calibration()
            except OSError as e:
                logging.error("Could not find the default calibration file for camera with serial number: "
                              + self.serial)
                return o3d.core.Tensor(np.identity(4), dtype=o3d.core.Dtype.Float32, device=DEVICE)
        return self._pose

    @property
    def pcd(self):
        """
        Access the point cloud form of the current frame.

        Returns:
            pcd(o3d.t.geometry.PointCloud): A tensor-based Open3D point cloud object.
        """
        if self._pcd.is_empty():
            self.compute_pcd(False)

        return self._pcd

    @property
    def intrinsics(self):
        """
        Access the camera's pinhole camera intrinsics.

        Returns:
            intrinsics (o3d.t.Tensor): 3x3 pinhole camera intrinsics matrix
        """
        return o3d.core.Tensor([[self._intrinsics.fx, 0, self._intrinsics.ppx],
                                [0, self._intrinsics.fy, self._intrinsics.ppy], [0, 0, 1]],
                               dtype=o3d.core.Dtype.Float32, device=DEVICE)

    @property
    def depth_scale(self):
        """
        Conversion factor to go from raw depth data to depth in meters. It is = to 0.001 by default for Realsense
        cameras. Warning: Open3D's depth_scale is = 1/depth_scale.

        Returns:
            depth_scale (float): scaling factor to get depth from Z16 to depth in meters.
        """
        if self._depth_scale is None:
            self._depth_scale = self._device.first_depth_sensor().get_depth_scale()

        return self._depth_scale

    @property
    def timestamp(self):
        """
        Access the timestamp of the latest frame

        Returns:
            timestamp(float): Timestamp of the most recent frame (ms), averaged of the two streams' timestamp.
        """

        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp):
        """
        Ensure timestamp is read-only.

        Args:
            timestamp: New timestamp
        """
        logging.warning("Tried to manually change the timestamp")
        pass

    @property
    def sensor_temp(self):
        """
        ASIC depth sensor temperature

        Returns:
            sensor_temp(float): sensor temperature
        """
        return self._device.first_depth_sensor().get_option(rs2.option.asic_temperature)

    @sensor_temp.setter
    def sensor_temp(self, temp):
        """
        Ensure temperature is read-only.

        Args:
            temp: New temperature value
        """
        logging.warning("Tried to manually change the sensor temperature")
        pass

    @property
    def projector_temp(self):
        """
        Projector depth sensor temperature

        Returns:
            projector_temp(float): projector temperature
        """
        return self._device.first_depth_sensor().get_option(rs2.option.projector_temperature)

    @projector_temp.setter
    def projector_temp(self, temp):
        """
        Ensure temperature is read-only.

        Args:
            temp: New temperature value
        """

        logging.warning("Tried to manually change the projector temperature")
        pass

    def seek(self, time_delta):
        """
        Args:
            time_delta:
        Returns:
        """
        self._playback.seek(time_delta)

    def compute_pcd(self, new_frame=False, depth_max=1.5):
        """
        Computes a Point Cloud from the newest acquired frame.
        Args:
            new_frame(bool): Bool to determine whether a new frame should be acquired first.
            depth_max(float): Depth threshold when computer the point cloud.
        Returns:
            pcd(o3d.t.Geometry.PointCloud): the point cloud representation of the newest acquired frame in an Open3D
                                            Tensor PointCloud format.
        """

        if new_frame:
            self.get_frames('o3d')

        if type(self.frame[0]) == rs2.depth_frame or type(self.frame[0]) == rs2.frame:
            self._rs_pc.map_to(self.frame[1])
            # Get the raw point cloud data
            dat = self._rs_pc.calculate(self.frame[0])
            # Changes how the data is "viewed" without making changes to the memory. Results in a list of 3D coordinates
            depth_dat = np.asanyarray(dat.get_vertices()).view(np.float32).reshape([-1, 3])
            # Reformat the colour data into a list of RGB values
            col_dat = np.asanyarray(self.frame[1].get_data()).reshape([-1, 3])
            col_t = o3d.core.Tensor(col_dat, device=DEVICE)
            depth_t = o3d.core.Tensor(depth_dat, device=DEVICE)
            self._pcd.point["positions"] = depth_t
            self._pcd.point["colors"] = col_t
        else:
            rgbdim = o3d.t.geometry.RGBDImage(self.frame[1], self.frame[0])
            self._pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbdim, self.intrinsics, depth_scale=1000,
                                                                         depth_max=depth_max)
        if self._pose is not None:
            self._pcd.transform(self._pose)

        return self._pcd

    def load_calibration(self, filename=None):
        """
        Initiates the pose attribute from a calibration file.
        Args:
            filename: String containing the filename of the calibration file.
        Returns:
            pose (numpy ndarray): a 4x4 matrix representing the camera to world transform
        """
        if filename is None:
            filename = self.serial + "_cal.txt"
        pose = np.loadtxt(filename)
        if not pose.shape == (4, 4):
            self.log.error(f"Error loading calibration: {calFile}. Does not conform to 4x4 matrix")
            return
        self._pose = o3d.core.Tensor(pose, device=DEVICE)

        return self._pose

    def waitStreaming(self, timeout=3):
        self._streamingFlag.wait(timeout=timeout)

    # Still needs thorough testing.
    def _rs_filter(self, rs_depth, filters):
        """
        Applies all the filter listed in the filters input argument. This function only works for realsense filters and
        the filter values are as defined in the class init method. Future development will allow for filter value to
        be added to a .json setting file.

        Args:
            rs_depth(rs2.depth_frame): Realsense depth frame to be filtered.
            filters(list(str)): List of filters to be applied.

        Returns:
            f_depth(rs2.depth_frame): Filtered Realsense frame

        """
        if filters is None:
            logging.warning("Tried to call a filter without specifying a filter")
            return
        f_depth = rs_depth
        for f in filters:
            f_depth = self._filters[f].process(f_depth)
        return f_depth

        # Encoding will be deprecated later, for now used for backward compatibility

    def get_frames(self, encoding='o3d', filtering=True, filters=['threshold'], timeout=5000):
        """
        Fetch an aligned depth and color frame.
        Args:
            encoding(List[char]): a flag that determines the type of the returned frame.
            filtering(bool): Filters are turned on if set to True
            filters(list(str)): Name of the filter to be applied.
        Returns:
            frame: a tuple with a depth and colour frame ordered (depth, colour). The type of the returned
        frames depend on the input value for encoding.
        """
        try:
            if not self._playback.current_status() == rs2.playback_status.stopped and self._frame is None:
                pass
            elif self._playback.current_status() == rs2.playback_status.stopped:
                # Raise a specific realsense error if get_frames was called before start_stream or after end_stream
                # self._pipe.get_active_profile()

                # If the above statement doesn't raise an exception, assume we reached the end of the file.
                raise NoMoreFrames("Reached the end of the stream")
        except AttributeError:
            pass

        frames = self._pipe.wait_for_frames(timeout_ms=timeout)

        aligned_frames = self._align_to_color.process(frames)

        rs_depth = aligned_frames.get_depth_frame()
        if filtering:
            rs_depth = self._rs_filter(rs_depth, filters)
        rs_color = aligned_frames.get_color_frame()
        self._timestamp = rs_depth.timestamp
        if encoding == 'o3d':
            np_depth = np.asanyarray(rs_depth.get_data())
            np_color = np.asanyarray(rs_color.get_data())
            depth = o3d.t.geometry.Image(o3d.core.Tensor(np_depth.astype(np.float32), device=DEVICE)).\
                filter_bilateral(7, 0.05, 10)
            color = o3d.t.geometry.Image(o3d.core.Tensor(np_color, device=DEVICE))
            self._frame = (depth, color)
        else:
            self._frame = (rs_depth, rs_color)

        # Reset the point cloud as it no longer represents the current frame. This will force calling of
        # compute_pcd method if the pcd attribute is accessed prior to a call to compute_pcd.
        self._pcd = o3d.t.geometry.PointCloud(device=DEVICE)
        return self._frame

    def get_o3d_intrinsics(self):
        """
        Transform the camera intrinsics to an open_3D format.
         Returns:
            intrinsics(o3d.camera.PinholeCameraIntrinsic): Return the intrinsics of the camera in an open3D format.
        """

        return o3d.camera.PinholeCameraIntrinsic(self._intrinsics.width, self._intrinsics.height, self._intrinsics.fx,
                                                 self._intrinsics.fy, self._intrinsics.ppx, self._intrinsics.ppy)

    def get_rs_intrinsics(self):
        """
            Transform the camera intrinsics to an open_3D format.
             Returns:
                intrinsics: Return the intrinsics of the camera in a realsense format.
        """

        return self._intrinsics

    def end_stream(self):
        """End the stream."""

        if not self.isStreaming:
            # not streaming
            return

        try:
            self.log.info("Stopping RS2 playback pipeline")
            device = self._pipe.get_active_profile().get_device()
            self._pipe.stop()

            #  There is a RealSense bug where the current status variable is not updated instantly. This resolves it.
            sleepDelay = 0.01
            maxTries = 5.0 / sleepDelay  # max delay of 5 sec
            tries = 0

            while tries < maxTries:
                time.sleep(sleepDelay)
                tries += 1

                if self._playback.current_status() == rs2.playback_status.stopped:
                    break

            self.log.info(f"Stopped: {device}")

        finally:
            # clear state vars
            self._streamingFlag.clear()

    def start_stream(self, offset=1, maxAttempts=5):
        """
        Starts or restarts the stream.
        Args:
            offset(float): Time in second to offset when encountering RealSense bug.
            maxAttempts(int): Number of attempts to resolve the RealSense bug.
        """

        if self.isStreaming:
            # already running
            self.log.warning("Aready streaming")
            return

        self.log.warning("Starting RS2 playback pipeline")

        # Specify that this device is from a captured stream
        self._pipe = rs2.pipeline()
        pipe_profile = self._pipe.start(self._cfg)
        self._device = pipe_profile.get_device()
        self._playback = self._device.as_playback()

        # Enable frame by frame access
        self._playback.set_real_time(False)

        # RealSense bug when accessing bag file is possible. The following section bypasses that bug.
        # Test whether frames any frames are received in 50 ms.
        temp = self._pipe.try_wait_for_frames(50)

        # If frame fetching failed or took longer than 50 ms.
        if not temp[0]:
            # Move the start of the recording by 1 sec
            self._playback.seek(datetime.timedelta(seconds=offset))
            if maxAttempts > 0:
                self.log.warning("Failed to start recording from the start, trying to restart with offset")
                # If frame fetching continues to fail.
                if not self._pipe.try_wait_for_frames(50)[0]:
                    self.log.warning("Failed to start recording from offset, trying to restart with larger offset")
                    self.log.warning("Setting offset to: " + str(offset + 0.5) + " sec")
                    self.start_stream(offset=offset + 0.5, maxAttempts=maxAttempts-1)
            else:
                return None

        # There is a second RealSense bug where the current status variable is not updated instantly. This resolves it.

        sleepDelay = 0.01
        maxTries = 5/sleepDelay  # maximum delay of 5 seconds.
        tries = 0
        while tries < maxTries:
            time.sleep(sleepDelay)
            tries += 1
            if self._playback.current_status() == rs2.playback_status.playing:
                delay = tries*sleepDelay
                self.log.warning("Delayed by " + str(tries) + " sec to allow stream to settle")
                break
        if self._playback.current_status() == rs2.playback_status.playing:
            self._streamingFlag.set()


class DualStream(Stream, ABC):
    """ Abstract class to handle a stream from two devices (2 cameras or 2 recordings)."""

    @abstractmethod
    def __init__(self, stream1, stream2):
        """ Initialise the dual stream."""

    @abstractmethod
    def get_frames(self):
        pass

    @property
    @abstractmethod
    def frame(self):
        pass

    @frame.setter
    def frame(self, new_frame):
        """
            Ensures that the frame attribute is only modified by fetching a new frame from the camera.
            Args:
                new_frame: New frame to replace existing frame.
        """
        self.log.warning('Manually setting the \'frame\' attribute is not allowed.')

    @property
    @abstractmethod
    def stream1(self):
        pass

    @stream1.setter
    def stream1(self, new_stream):
        """
            Ensures that the stream is only set during initialization.
        Args:
            new_stream: New stream to replace the existing stream1.
        """
        logging.warning('Tried to reset \'stream1\', stream attributes can only be set during initialization.')

    @property
    @abstractmethod
    def stream2(self):
        pass

    @stream2.setter
    def stream2(self, new_stream):
        """
            Ensures that the stream is only set during initialization.
        Args:
            new_stream: New stream to replace the existing stream1.
        """
        logging.warning('Tried to reset \'stream2\', stream attributes can only be set during initialization.')

    @property
    @abstractmethod
    def pose(self):
        pass

    @abstractmethod
    def load_calibration(self, filename=None):
        pass

    @abstractmethod
    def start_stream(self):
        pass

    @abstractmethod
    def end_stream(self):
        pass


class DualRecording(DualStream):
    """Class to handle the interaction with the Realsense API when handling bag file recordings from 2 cameras."""

    def __init__(self, stream1, stream2, cal_file=None):
        """
        Initializes the DualRecording object by setting both stream property
        Args:
            stream1: First stream of the dual stream.
            stream2: Second stream of the dual stream.
            cal_file(list:string): list of filepath to the calibration files
        """
        # Initialising frame attribute.
        self._frame = None

        # Timestamp of the most current depth frame.
        self._timestamp = None

        if cal_file is not None:
            self.load_calibration(cal_file)
        else:
            self._pose = None

        # A point cloud object representing the latest frame. Stored in a Open3D Tensor PointCloud format.
        self._pcd = o3d.t.geometry.PointCloud(device=DEVICE)

        # Check if input is a recording object or a filepath to a recording
        if type(stream1) == Recording:
            self._stream1 = stream1
        else:
            try:
                self._stream1 = Recording(stream1)
            except RuntimeError:
                logging.error('Could not resolve the input for \'stream1\'.')
                raise ValueError('Could not resolve the input for \'stream1\'.' )
        if type(stream2) == Recording:
            self._stream2 = stream2
        else:
            try:
                self._stream2 = Recording(stream2)
            except RuntimeError:
                logging.error('Could not resolve the input for \'stream2\'.')
                raise ValueError('Could not resolve the input for \'stream2\'.')

    @property
    def stream1(self):
        """
            Access the first stream.
        Returns:
            stream1: The first stream.
        """
        return self._stream1

    @property
    def stream2(self):
        """
            Access the second stream.
        Returns:
            stream2: The first stream.
        """
        return self._stream2

    @property
    def frame(self):
        """
        Access the last frame.
        Returns:
            frame: A tuple containing the last frame ordered (depth, colour)
        """
        if self._frame is None:
            self._frame = self.get_frames()
        return self._frame

    @property
    def pose(self):
        """
        Access the 4x4 matrices representing the cameras' poses.
        Returns:
            pose (numpy ndarray): a tuple of 4x4 matrix representing the cameras to world transform
        """

        if self._pose is None:
            try:  # Try because we're calling load_calibration with the default filename
                self.load_calibration()
            except IOError:  # Need to doublecheck that this is the right exception.
                pass  # Need to ensure proper error message is raised here.
        return self._pose

    @property
    def pcd(self):
        """
        Access the point cloud form of the current frame.
        Returns:
            pcd: A tensor-based Open3D point cloud object.
        """
        if self._pcd.is_empty():
            self.compute_pcd()

        return self._pcd

    @pcd.setter
    def pcd(self, pcd):
        """
        Ensures pcd is read-only
        Args:
            pcd: New point cloud
        """
        logging.warning("tried to manually set a new point cloud")
        pass

    @property
    def timestamp(self):
        """
        Access the timestamp of the latest frame
        Returns:
            timestamp: Timestamp of the most recent frame (ms), averaged of the two streams' timestamp.
        """

        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp):
        """
        Ensure timestamp is read-only.
        Args:
            timestamp: New timestamp
        """
        logging.warning("Tried to manually change the timestamp")
        pass

    def seek(self, time_delta):
        """
        Args:
            time_delta:
        Returns:
        """
        self.stream1.seek(time_delta)
        self.stream2.seek(time_delta)

    def compute_pcd(self, new_frame=False):
        """
        Computes a Point Cloud from the newest acquired frame.
        Args:
            new_frame: Bool to determine whether a new frame should be acquired first.
        Returns:
            pcd: the point cloud representation of the newest acquired frame in an Open3D Tensor PointCloud format.
        """

        if new_frame:
            self.get_frames()

        if self._pose is None:
            if self._stream1.pose is None or self._stream2.pose is None:
                self.log.warning("Tried to compute a dual point cloud of a stream that has not been calibrated.")
                try:
                    self.load_calibration()
                except DualStreamError:
                    raise DualStreamError("Cannot compute the point cloud of a dual stream that has not been "
                                          "calibrated.")
            else:
                self._pose = (self._stream1.pose, self._stream2.pose)
        pcd1 = self.stream1.compute_pcd(new_frame)
        pcd2 = self.stream2.compute_pcd(new_frame)

        self._pcd = pcd1 + pcd2
        return self._pcd

    def load_calibration(self, filename=None, force=False):
        """
        Uses each stream's load_calibration method to initiate their pose attribute.
        Args:
            filename: list of two strings containing the filename of the calibration files.
            force(bool): If True, will always prioritise calibration in file. Otherwise, prioritise pre-existing
                        calibration
        """

        if filename is not None:
            if len(filename) == 2:
                pose1 = self.stream1.load_calibration(filename[0])
                pose2 = self.stream2.load_calibration(filename[1])
            else:
                logging.warning("load_calibration method of a DualRecording object can only accept a list of exactly 2 "
                                "files")
                try:
                    pose1 = self.stream1.load_calibration()
                    pose2 = self.stream2.load_calibration()
                except OSError:
                    logging.error("Could not complete calibration with default filename for DualStream object")
                    raise DualStreamError("Tried to complete default calibration but could not find calibration files")
        else:
            if force is False:
                try:
                    if self.stream1.pose is not None:
                        pose1 = self.stream1.pose
                    else:
                        pose1 = self.stream1.load_calibration()
                    if self.stream2.pose is not None:
                        pose2 = self.stream2.pose
                    else:
                        pose2 = self.stream2.load_calibration()
                except OSError:
                    logging.error("Could not complete calibration with default filename for DualStream object")
                    raise DualStreamError("Tried to complete default calibration but could not find calibration files")
            else:
                try:
                    pose1 = self.stream1.load_calibration()
                    pose2 = self.stream2.load_calibration()
                except OSError:
                    logging.error("Could not complete calibration with default filename for DualStream object")
                    raise DualStreamError("Tried to complete default calibration but could not find calibration files")

        self._pose = (pose1, pose2)
        return self._pose

    def get_frames(self, encoding='o3d', filtering=True, filters=["threshold"]):
        """
        Fetch new frame and ensure temporal alignment.

        Args:
            encoding(List[char]): a flag that determines the type of the returned frame.
            filtering(bool): Filters are turned on if set to True
            filters(List[str]): List of filters to be applied to the newly fetched frames

        Returns:
            frame(Tuple): A tuple with a frame from each of the camera. The type of the returned frames depend on the
                          input value for encoding.

        """
        # Getting new frames with a call  to the cameras.
        f1 = self._stream1.get_frames(encoding=encoding, filtering=filtering, filters=filters)
        f2 = self._stream2.get_frames(encoding=encoding, filtering=filtering, filters=filters)

        # Get the delay between the depth frames of each camera
        diff = self.stream2.timestamp - self.stream1.timestamp
        # If delay is more than 30ms, figure out which camera is behind and call new frames from that camera until
        # Both cameras are synchronised.
        while np.abs(diff) > 100:
            if diff > 100:
                print("Timestamps:" + str(self.stream1.timestamp) + " " + str(self.stream2.timestamp))
                f1 = self._stream1.get_frames(encoding=encoding, filtering=filtering)
                diff = self.stream2.timestamp - self.stream1.timestamp
                continue
            elif diff < -100:
                print("Timestamps:" + str(self.stream1.timestamp) + " " + str(self.stream2.timestamp))
                f2 = self._stream2.get_frames(encoding=encoding, filtering=filtering)
                diff = self.stream2.timestamp - self.stream1.timestamp
                continue

        self._timestamp = (self.stream1.timestamp + self.stream2.timestamp)/2
        self._frame = (f1, f2)

        # Reset the point cloud as it no longer represents the current frame. This will force calling of
        # compute_pcd method if the pcd attribute is accessed prior to a call to compute_pcd.
        self._pcd = o3d.t.geometry.PointCloud(device=DEVICE)
        return self._frame

    def start_stream(self):
        """Starts or restarts the stream by starting both stream1 and stream2."""

        self._stream1.start_stream()
        self._stream2.start_stream()
        time.sleep(0.0001)

    def end_stream(self):
        """Ends the stream by ending both stream1 and stream2."""

        self._stream1.end_stream()
        self._stream2.end_stream()


class DualCamera(DualStream):
    """Class to handle the interaction with the Realsense API when handling realtime streams from 2 cameras."""

    def __init__(self, stream1, stream2, cal_file=None):
        """
        Initialises the DualCamera object by setting both stream properties
        Args:
            stream1(spygrt.stream.Camera): First stream of the dual stream.
            stream2(spygrt.stream.Camera): Second stream of the dual stream.
            cal_file(list:string): list of filepath to the calibration files
        """
        # Initialising frame attribute.
        self._frame = None

        # Timestamp of the most current depth frame.
        self._timestamp = None

        if cal_file is not None:
            self.load_calibration(cal_file)
        else:
            self._pose = None

        # A point cloud object representing the latest frame. Stored in a Open3D Tensor PointCloud format.
        self._pcd = o3d.t.geometry.PointCloud(device=DEVICE)

        # Check if input is a recording object or a filepath to a recording
        if type(stream1) == Camera:
            self._stream1 = stream1
        else:
            try:
                self._stream1 = Camera(stream1, inter_cam_sync_mode=1)
            except RuntimeError:
                logging.error('Could not resolve the input for \'stream1\'.')
                raise ValueError('Could not resolve the input for \'stream1\'.' )
        if type(stream2) == Camera:
            self._stream2 = stream2
        else:
            try:
                self._stream2 = Camera(stream2, inter_cam_sync_mode=2)
            except RuntimeError:
                logging.error('Could not resolve the input for \'stream2\'.')
                raise ValueError('Could not resolve the input for \'stream2\'.')
    @property
    def stream1(self):
        """
            Access the first stream.
        Returns:
            stream1: The first stream.
        """
        return self._stream1

    @property
    def stream2(self):
        """
            Access the second stream.
        Returns:
            stream2: The first stream.
        """
        return self._stream2

    @property
    def frame(self):
        """
        Access the last frame.
        Returns:
            frame: A tuple containing the last frame ordered (depth, colour)
        """
        if self._frame is None:
            self._frame = self.get_frames()
        return self._frame

    @property
    def pose(self):
        """
        Access the 4x4 matrices representing the cameras' poses.
        Returns:
            pose (numpy ndarray): a tuple of 4x4 matrix representing the cameras to world transform
        """

        if self._pose is None:
            try:  # Try because we're calling load_calibration with the default filename
                self.load_calibration()
            except IOError:  # Need to doublecheck that this is the right exception.
                pass  # Need to ensure proper error message is raised here.
        return self._pose

    @property
    def pcd(self):
        """
        Access the point cloud form of the current frame.
        Returns:
            pcd: A tensor-based Open3D point cloud object.
        """
        if self._pcd.is_empty():
            self.compute_pcd()

        return self._pcd

    @pcd.setter
    def pcd(self, pcd):
        """
        Ensures pcd is read-only
        Args:
            pcd: New point cloud
        """
        logging.warning("tried to manually set a new point cloud")
        pass

    @property
    def timestamp(self):
        """
        Access the timestamp of the latest frame
        Returns:
            timestamp: Timestamp of the most recent frame (ms), averaged of the two streams' timestamp.
        """

        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp):
        """
        Ensure timestamp is read-only.
        Args:
            timestamp: New timestamp
        """
        logging.warning("Tried to manually change the timestamp")
        pass

    def compute_pcd(self, new_frame=False):
        """
        Computes a Point Cloud from the newest acquired frame.
        Args:
            new_frame: Bool to determine whether a new frame should be acquired first.
        Returns:
            pcd: the point cloud representation of the newest acquired frame in an Open3D Tensor PointCloud format.
        """

        if new_frame:
            self.get_frames()

        if self._pose is None:
            if self._stream1.pose is None or self._stream2.pose is None:
                logging.warning("Tried to compute a dual point cloud of a stream that has not been calibrated.")
                try:
                    self.load_calibration()
                except DualStreamError:
                    raise DualStreamError("Cannot compute the point cloud of a dual stream that has not been "
                                          "calibrated.")
            else:
                self._pose = (self._stream1.pose, self._stream2.pose)
        pcd1 = self.stream1.compute_pcd(new_frame)
        pcd2 = self.stream2.compute_pcd(new_frame)

        self._pcd = pcd1 + pcd2
        return self._pcd

    def load_calibration(self, filename=None, force=False):
        """
        Uses each stream's load_calibration method to initiate their pose attribute.
        Args:
            filename: list of two strings containing the filename of the calibration files.
            force(bool): If True, will always prioritise calibration in file. Otherwise, prioritise pre-existing
                        calibration
        """

        if filename is not None:
            if len(filename) == 2:
                pose1 = self.stream1.load_calibration(filename[0])
                pose2 = self.stream2.load_calibration(filename[1])
            else:
                logging.warning("load_calibration method of a DualRecording object can only accept a list of exactly 2 "
                                "files")
                try:
                    pose1 = self.stream1.load_calibration()
                    pose2 = self.stream2.load_calibration()
                except OSError:
                    logging.error("Could not complete calibration with default filename for DualStream object")
                    raise DualStreamError("Tried to complete default calibration but could not find calibration files")
        else:
            if force is False:
                try:
                    if self.stream1.pose is not None:
                        pose1 = self.stream1.pose
                    else:
                        pose1 = self.stream1.load_calibration()
                    if self.stream2.pose is not None:
                        pose2 = self.stream2.pose
                    else:
                        pose2 = self.stream2.load_calibration()
                except OSError:
                    logging.error("Could not complete calibration with default filename for DualStream object")
                    raise DualStreamError("Tried to complete default calibration but could not find calibration files")
            else:
                try:
                    pose1 = self.stream1.load_calibration()
                    pose2 = self.stream2.load_calibration()
                except OSError:
                    logging.error("Could not complete calibration with default filename for DualStream object")
                    raise DualStreamError("Tried to complete default calibration but could not find calibration files")

        self._pose = (pose1, pose2)
        return self._pose

    def get_frames(self, encoding='o3d', filtering=True, filters=["threshold"]):
        """
        Fetch new frame and ensure temporal alignment.

        Args:
            encoding(List[char]): a flag that determines the type of the returned frame.
            filtering(bool): Filters are turned on if set to True
            filters(List[str]): List of filters to be applied to the newly fetched frames

        Returns:
            frame(Tuple): A tuple with a frame from each of the camera. The type of the returned frames depend on the
                          input value for encoding.

        """
        # Getting new frames with a call  to the cameras.
        f1 = self._stream1.get_frames(encoding=encoding, filtering=filtering, filters=filters)
        f2 = self._stream2.get_frames(encoding=encoding, filtering=filtering, filters=filters)

        # This section will be obsolete when inter cam sync is enabled.

        # # Get the delay between the depth frames of each camera
        # diff = f2[0].timestamp - f1[0].timestamp
        # # If delay is more than 30ms, figure out which camera is behind and call new frames from that camera until
        # # Both cameras are synchronised.
        # while np.abs(diff) > 100:
        #     if diff > 100:
        #         print("Timestamps:" + str(f1[0].timestamp) + " " + str(f1[0].frame_number) + " "
        #               + str(f2[0].timestamp) + " " + str(f2[0].frame_number))
        #         f1 = self._stream1.get_frames(encoding='rs')
        #         diff = f2[0].timestamp - f1[0].timestamp
        #         continue
        #     elif diff < -100:
        #         print("Timestamps:" + str(f1[0].timestamp) + " " + str(f1[0].frame_number) + " "
        #               + str(f2[0].timestamp) + " " + str(f2[0].frame_number))
        #         f2 = self._stream2.get_frames(encoding='rs')
        #         diff = f2[0].timestamp - f1[0].timestamp
        #         continue

        self._timestamp = (self.stream1.timestamp + self.stream2.timestamp)/2
        self._frame = (f1, f2)

        # Reset the point cloud as it no longer represents the current frame. This will force calling of
        # compute_pcd method if the pcd attribute is accessed prior to a call to compute_pcd.
        self._pcd = o3d.t.geometry.PointCloud(device=DEVICE)
        return self._frame

    def start_stream(self, rec=False, bag_file=None):
        """
        Starts or restarts the stream by starting both stream1 and stream2.
        Args:
            rec(bool): Indicates whether a recording should be created
            bag_file(list:string): Filepath to where the bag_files should be recorded
        """
        if rec:
            if bag_file is None:
                self._stream1.start_stream(rec=rec)
                self._stream2.start_stream(rec=rec)
            else:
                try:
                    self._stream1.start_stream(rec=rec, bag_file=bag_file[0])
                    self._stream2.start_stream(rec=rec, bag_file=bag_file[1])
                except TypeError:
                    self._stream1.start_stream(rec=rec)
                    self._stream2.start_stream(rec=rec)

        else:
            self._stream1.start_stream()
            self._stream2.start_stream()

        time.sleep(0.0001)

    def end_stream(self):
        """Ends the stream by ending both stream1 and stream2."""

        self._stream1.end_stream()
        self._stream2.end_stream()