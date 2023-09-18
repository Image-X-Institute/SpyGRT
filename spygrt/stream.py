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


# # Setting up a logger that will write log messages in the 'stream.log' file.
# logger = logging.getLogger(__name__)
# fh = logging.FileHandler('stream.log', 'w', 'utf-8')
# fh.setLevel(logging.DEBUG)
# formatter = logging.Formatter('- %(module)s - %(levelname)-8s: %(message)s')
# fh.setFormatter(formatter)
# logger.addHandler(fh)

DEVICE = None
ALLOW_GPU = os.environ.get('_RTM_GUI_ALLOWGPU')

if hasattr(o3d, 'cuda') and ALLOW_GPU=="1":
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

class StreamBase(Stream):
    """Base class to wrap common pipeline properties"""
    def __init__(self, warmupFrames=50):
        # logger
        self.log = logging.getLogger(__name__)

        # number of frames to allow for warmup
        self._warmupFrames = warmupFrames

        # flag to indicate stream is active and warmed up or closed down properly
        self._streamingFlag = threading.Event()

    @property
    def isStreaming(self):
        return self._streamingFlag.is_set()

    def waitStreaming(self, timeout=3):
        self._streamingFlag.wait(timeout=timeout)
class Camera(StreamBase):
    """Class to handle the interaction with the Realsense API for real-time frame capture"""

    def __init__(self, device, cal_file=None):
        """
        Initialise the camera settings and start the stream.
        Args:
            device(rs2.device): Realsense device object pointing to a camera(rs2.device).
            cal_file(string): Filepath to the calibration file
        """
        
        super().__init__()

        # Initialising frame attribute.
        self._frame = None
        self._device = device

        if cal_file is not None:
            self.load_calibration(cal_file)
        else:
            self._pose = None

        # Realsense point cloud object useful for the compute_pcd method. For internal class use only.
        self._rs_pc = rs2.pointcloud()

        # Realsense threshold filter object for processing frames.
        self._threshold_filter = None#rs2.threshold_filter(0.1, 5)#(0.1, 1.3)

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
            depth_sensor.set_option(rs2.option.emitter_enabled, True)

        if depth_sensor.supports(rs2.option.laser_power):
            laser = depth_sensor.get_option_range(rs2.option.laser_power)
            depth_sensor.set_option(rs2.option.laser_power, laser.max)

        try:
            depth_sensor.set_option(rs2.option.max_distance, 200)
        except RuntimeError:
            self.log.info("no maximum distance setting for camera model: " + self.model)

        try:
            depth_sensor.set_option(rs2.option.min_distance, 0)
        except RuntimeError:
            self.log.info("no minimum distance setting for camera model: " + self.model)

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
            pose (numpy ndarray): a 4x4 matrix representing the camera to world transform
        """

        if self._pose is None:
            try:  # Try because we're calling load_calibration without a filename and using the default filename and
                # location
                self.load_calibration()
            except OSError as e:
                self.log.error("Could not find the default calibration file for camera with serial number: "
                              + self.serial)
                return o3d.core.Tensor(np.identity(4), dtype=o3d.core.Dtype.Float32, device=DEVICE).cpu().numpy()
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
        logger.warning("tried to manually set a new point cloud")
        pass

    @property
    def intrinsics(self):
        """
            Access the camera's pinhole camera intrinsics.
            Returns:
                intrinsics (o3d.t.Tensor): 3x3 pinhole camera intrinsics matrix
        """
        return o3d.core.Tensor([[self._intrinsics.fx, 0, self._intrinsics.ppx],
                                [0, self._intrinsics.fy, self._intrinsics.ppy], [0, 0, 1]], device=DEVICE)

    @property
    def depth_scale(self):
        """
            Conversation factor to go from raw depth data to depth in meters. It is = to 0.001 by default for
            Realsense cameras. Warning: Open3D's depth_scale is = 1/depth_scale.
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
        logger.warning("Tried to manually change the timestamp")
        pass

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
            self.get_frames('rs')

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
                                                                         depth_max=1.5)

        if self._pose is not None:
            self._pcd.transform(o3d.core.Tensor(self._pose, device=DEVICE))

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
        self._pose = np.loadtxt(filename)

        return self._pose

    # Encoding will be deprecated later, for now used for backward compatibility
    def get_frames(self, encoding='rs', timeout=5000):
        """
        Fetch an aligned depth and color frame.
        Args:
            encoding: a flag that determines the type of the returned frame.
        Returns:
            frame: a tuple with a depth and colour frame ordered (depth, colour). The type of the returned
        frames depend on the input value for encoding.
        """
        """"""
        if not self.warmed:
            self.warmup()
        frames = self._pipe.wait_for_frames(timeout_ms=timeout)

        aligned_frames = self.__align_to_color.process(frames)

        rs_depth = aligned_frames.get_depth_frame()
        rs_color = aligned_frames.get_color_frame()
        # Computationally intensive depth image filtering
        # rs_depth = self._spatial_filter.process(rs_depth)

        # apply thresholding if defined
        if self._threshold_filter is not None:
            rs_depth = self._threshold_filter.process(rs_depth)

        self._timestamp = rs_depth.timestamp
        if encoding == 'o3d':
            np_depth = np.asanyarray(rs_depth.get_data())
            np_color = np.asanyarray(rs_color.get_data())
            depth = o3d.t.geometry.Image(o3d.core.Tensor(np_depth, device=DEVICE))
            color = o3d.t.geometry.Image(o3d.core.Tensor(np_color, device=DEVICE))
            self._frame = (depth, color)

        else:
            self._frame = (rs_depth, rs_color)
        return self._frame

    def warmup(self):
        """"Warmup the camera before frame acquisition"""
        for i in range(self._warmupFrames):
            try:
                self._pipe.wait_for_frames()

            except Exception as e:
                raise Exception(f"{e}. Please check camera is not connected to another application.")

        self.log.info("Warmed up")

        self.warmed = True
        self._streamingFlag.set()

        # self.intrinsics = frames.get_depth_frame().get_profile().as_video_stream_profile().intrinsics

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


class Recording(StreamBase):
    """Class to handle the interaction with the Realsense API for realsense bag files."""

    def __init__(self, device, filename=None, repeat=False, ctx=None):
        """
        Args:
            device: link to a bag file containing a recording.
        """

        super().__init__()

        self._ctx = ctx # re-use provided context

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

        # A point cloud object representing the latest frame. Stored in a Open3D Tensor PointCloud format.
        self._pcd = o3d.t.geometry.PointCloud(device=DEVICE)

        # Initialise config object
        self._cfg = rs2.config()

        # Configure the camera with the default setting.
        self._cfg.enable_device_from_file(device, repeat)

        # re-use given context if provided (this is necessary to avoid COM errors in RTMGUI)
        ctx = rs2.context() if self._ctx is None else self._ctx

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
            pose (numpy ndarray): a 4x4 matrix representing the camera to world transform
        """

        if self._pose is None:
            try:  # Try because we're calling load_calibration without a filename and using the default filename and
                # location
                self.load_calibration()
            except OSError as e:
                self.log.error("Could not find the default calibration file for camera with serial number: "
                              + self.serial)
                raise e
        return self._pose

    @property
    def pcd(self):
        """
        Access the point cloud form of the current frame.
        Returns:
            pcd: A tensor-based Open3D point cloud object.
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
                                [0, self._intrinsics.fy, self._intrinsics.ppy], [0, 0, 1]], device=DEVICE)

    @property
    def depth_scale(self):
        """
            Conversation factor to go from raw depth data to depth in meters. It is = to 0.001 by default for
            Realsense cameras. Warning: Open3D's depth_scale is = 1/depth_scale.
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
        logger.warning("Tried to manually change the timestamp")
        pass

    def seek(self, time_delta):
        """
        Args:
            time_delta:
        Returns:
        """
        self._playback.seek(time_delta)

    def compute_pcd(self, new_frame=False):
        """
        Computes a Point Cloud from the newest acquired frame.
        Args:
            new_frame: Bool to determine whether a new frame should be acquired first.
        Returns:
            pcd: the point cloud representation of the newest acquired frame in an Open3D Tensor PointCloud format.
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
                                                                         depth_max=1.5)
        if self._pose is not None:
            self._pcd.transform(o3d.core.Tensor(self._pose, device=DEVICE))

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
        self._pose = np.loadtxt(filename)

        return self._pose

    def get_frames(self, encoding='o3d', timeout=5000):
        """
        Fetch a new set of depth and colour frame.
        Args:
            encoding: a flag that determines the type of the returned frame.
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
                # raise NoMoreFrames("Reached the end of the stream")
                pass
        except AttributeError:
            pass

        frames = self._pipe.wait_for_frames(timeout_ms=timeout)

        aligned_frames = self._align_to_color.process(frames)

        rs_depth = aligned_frames.get_depth_frame()
        rs_depth = self._threshold_filter.process(rs_depth)
        # Computationally intensive depth image filtering
        # rs_depth = self._spatial_filter.process(rs_depth)
        rs_color = aligned_frames.get_color_frame()
        self._timestamp = rs_depth.timestamp
        if encoding == 'o3d':
            np_depth = np.asanyarray(rs_depth.get_data())
            np_color = np.asanyarray(rs_color.get_data())
            depth = o3d.t.geometry.Image(o3d.core.Tensor(np_depth.astype(np.float32), device=DEVICE)).filter_bilateral(7, 0.05, 10)
            color = o3d.t.geometry.Image(o3d.core.Tensor(np_color, device=DEVICE))
            self._frame = (depth, color)
        else:
            self._frame = (rs_depth, rs_color)

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

            sleepDelay = 0.01
            maxTries = 5.0/sleepDelay # max delay of 5 sec
            tries = 0

            while tries<maxTries:
                time.sleep(sleepDelay)
                tries += 1
                
                if self._playback.current_status() == rs2.playback_status.stopped:
                    break

            self.log.info(f"Stopped: {device}")

        finally:
            # clear state vars
            self._streamingFlag.clear()

    def start_stream(self, offset=1):
        """Starts or restarts the stream."""

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
        temp = self._pipe.try_wait_for_frames(self._warmupFrames)

        if not temp[0]:
            i = 1

            self._playback.seek(datetime.timedelta(seconds=i))

            if not self._pipe.try_wait_for_frames(50)[0]:
                self.log.warning("Trying to restart with offset")
                self.start_stream(offset=offset + 0.5)

        sleepDelay = 0.01 
        maxTries = 5.0/sleepDelay # max delay of 5 sec
        tries = 0

        while tries<maxTries:
            time.sleep(sleepDelay)
            tries += 1
            
            if self._playback.current_status() == rs2.playback_status.playing:
                break

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
        logging.warning('Manually setting the \'frame\' attribute is not allowed.')

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
        # init logger here
        self.log = logging.getLogger(__name__)

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
        logger.warning("tried to manually set a new point cloud")
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
        logger.warning("Tried to manually change the timestamp")
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

    def get_frames(self, encoding='o3d', timeout=5000):
        """
            Fetch new frame and ensure temporal alignment.
        Returns:
            frame:Tuple containing two sets of (depth,colour) frames, 1 for each stream.
        """
        # Getting new frames with a call  to the cameras.
        f1 = self._stream1.get_frames(encoding=encoding, timeout=timeout)
        f2 = self._stream2.get_frames(encoding=encoding, timeout=timeout)

        # Get the delay between the depth frames of each camera
        diff = self.stream2.timestamp - self.stream1.timestamp
        # If delay is more than 30ms, figure out which camera is behind and call new frames from that camera until
        # Both cameras are synchronised.
        while np.abs(diff) > 100:
            if diff > 100:
                self.log.debug("SYNC :: Timestamps:" + str(self.stream1.timestamp) + " " + str(self.stream2.timestamp))
                f1 = self._stream1.get_frames(encoding=encoding, timeout=timeout)
                diff = self.stream2.timestamp - self.stream1.timestamp
                continue
            elif diff < -100:
                self.log.debug("SYNC :: Timestamps:" + str(self.stream1.timestamp) + " " + str(self.stream2.timestamp))
                f2 = self._stream2.get_frames(encoding=encoding, timeout=timeout)
                diff = self.stream2.timestamp - self.stream1.timestamp
                continue

        self._timestamp = (self.stream1.timestamp + self.stream2.timestamp)/2
        self._frame = (f1, f2)
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
        # init logger here
        self.log = logging.getLogger(__name__)

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
                self._stream1 = Camera(stream1)
            except RuntimeError:
                logging.error('Could not resolve the input for \'stream1\'.')
                raise ValueError('Could not resolve the input for \'stream1\'.' )
        if type(stream2) == Camera:
            self._stream2 = stream2
        else:
            try:
                self._stream2 = Camera(stream2)
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
        logger.warning("tried to manually set a new point cloud")
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
        logger.warning("Tried to manually change the timestamp")
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

    def load_calibration(self, filename=None):
        """
        Uses each stream's load_calibration method to initiate their pose attribute.
        Args:
            filename: list of two strings containing the filename of the calibration files.
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

        self._pose = (pose1, pose2)
        return self._pose

    def get_frames(self, encoding='o3d', timeout=5000):
        """
            Fetch new frame and ensure temporal alignment.
        Returns:
            frame:Tuple containing two sets of (depth,colour) frames, 1 for each stream.
        """
        # Getting new frames with a call  to the cameras.
        f1 = self._stream1.get_frames(encoding=encoding, timeout=timeout)
        f2 = self._stream2.get_frames(encoding=encoding, timeout=timeout)

        # There should be no delays for Cameras, so this section may be obsolete:

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



class StreamOld:
    """Parent Class to handle the interactions with the RealSense API"""

    def __init__(self, stream_origin):
        """Initialise the camera settings and start the stream"""

        # pipeline that controls the acquisition of frames.
        self.__pipe = rs2.pipeline()

        # Initialise config object
        cfg = self._get_config(stream_origin)

        # Start the pipeline - Frame fetching is possible after this.
        pipe_profile = self.__pipe.start(cfg)

        # Object that align depth and color.
        self.__align_to_color = rs2.align(rs2.stream.color)

        # Device object to access metadata.
        self.__device = pipe_profile.get_device()

        # Serial number of the camera that captured this stream.
        self.serial = self.__device.get_info(rs2.camera_info.serial_number)

        # Model of the camera that captured this stream.
        self.model = self.__device.get_info(rs2.camera_info.name)

        # Access unit in which the depth is saved
        depth_sensor = self.__device.first_depth_sensor()
        self.depthscale = depth_sensor.get_depth_scale()

        # Internal variable to ensure the first frame isn't lost.
        self.__first = True
        # Set up the intrinsics.
        self.frame = self.get_frames()

        # Pinhole camera intrinsics for depth camera
        self.intrinsics = depth_sensor.get_stream_profiles()[0].as_video_stream_profile().intrinsics

    def get_frames(self):
        """Fetch an aligned depth and color frame."""
        frames = self.__pipe.wait_for_frames()
        aligned_frames = self.__align_to_color.process(frames)
        rs_depth = aligned_frames.get_depth_frame()
        rs_color = aligned_frames.get_color_frame()
        if self.__first:
            self.intrinsics = rs_depth.get_profile().as_video_stream_profile().intrinsics
            self.__first = False
        np_depth = np.float32(rs_depth.get_data()) * 1 / 65535
        np_color = np.asanyarray(rs_color.get_data())

        depth = o3d.t.geometry.Image(o3d.core.Tensor(np_depth))
        color = o3d.t.geometry.Image(o3d.core.Tensor(np_color))
        self.frame = (depth, color)
        return self.frame

    def get_o3d_intrinsics(self):
        """Transform the camera intrinsics to an open_3D format."""
        return o3d.camera.PinholeCameraIntrinsic(self.intrinsics.width, self.intrinsics.height, self.intrinsics.fx,
                                                 self.intrinsics.fy, self.intrinsics.ppx, self.intrinsics.ppy)

    def end_stream(self):
        """Ends the stream."""
        self.__pipe.stop()

    def _get_config(self, stream_origin):
        """Set up configuration for the stream."""
        return rs2.config()


class RecordingOld(StreamOld):
    """Class to handle the interaction with the Realsense API for realsense bag files"""

    def __init__(self, bag_file):
        """Initialise playback setting and start the stream"""

        StreamOld.__init__(self, bag_file)

        # Specify that this device is from a captured stream
        self.__playback = self._StreamOld__device.as_playback()

        # Enable frame by frame access
        self.__playback.set_real_time(False)

    def _get_config(self, bag_file):
        """Set up configuration for the stream."""
        cfg = rs2.config()
        # Configure the camera with the default setting.
        cfg.enable_device_from_file(bag_file, False)
        return cfg

    def get_frames(self):
        """return an aligned depth and color frame"""
        try:
            if not self.__playback.current_status() == rs2.playback_status.stopped and self._StreamOld__first:
                self._StreamOld__first = False
                return self.frame
            elif self.__playback.current_status() == rs2.playback_status.stopped:
                raise NoMoreFrames("Reached the end of the stream")
        except AttributeError:
            pass

        frames = self._StreamOld_pipe.wait_for_frames()
        aligned_frames = self._StreamOld__align_to_color.process(frames)
        rs_depth = aligned_frames.get_depth_frame()
        rs_color = aligned_frames.get_color_frame()

        if self._StreamOld__first:
            self.intrinsics = rs_depth.get_profile().as_video_stream_profile().intrinsics

        np_depth = np.float32(rs_depth.get_data()) * 1 / 65535
        np_color = np.asanyarray(rs_color.get_data())

        depth = o3d.t.geometry.Image(o3d.core.Tensor(np_depth))
        color = o3d.t.geometry.Image(o3d.core.Tensor(np_color))

        self.frame = (depth, color)
        return self.frame


class CameraOld(StreamOld):
    """Class to handle the interaction with the Realsense API for real-time frame capture"""

    def __init__(self, device):
        """Initialise the camera settings and start the stream"""

        # Initialise serial number early as it is needed for Stream configuration
        self.serial = device.get_info(rs2.camera_info.serial_number)

        # Indicate whether the camera has warmed up. If false, need to leave the camera running
        # for 10-15frames before any get_frames can be used.
        self.warmedup = False

        StreamOld.__init__(self, device)

        depth_sensor = self._StreamOld__device.first_depth_sensor()

        if depth_sensor.supports(rs2.option.emitter_enabled):
            depth_sensor.set_option(rs2.option.emitter_enabled, True)

        if depth_sensor.supports(rs2.option.laser_power):
            laser = depth_sensor.get_option_range(rs2.option.laser_power)
            depth_sensor.set_option(rs2.option.laser_power, laser.max)

        try:
            depth_sensor.set_option(rs2.option.max_distance, 200)
        except:
            print("no maximum distance setting for camera model: " + self.model)

        try:
            depth_sensor.set_option(rs2.option.min_distance, 0)
        except:
            print("no minimum distance setting for camera model: " + self.model)

        # Warmup
        self.warmup()

    def warmup(self):
        """Warmup the camera before frame acquisition"""
        for i in range(30):
            frames = self._StreamOld__pipe.wait_for_frames()
        self.warmedup = True
        self._Stream__first = False
        self.intrinsics = frames.get_depth_frame().get_profile().as_video_stream_profile().intrinsics

    def get_frames(self):
        """Fetch an aligned depth and color frame"""

        if not self.warmedup:
            self.warmup()
        frames = self._StreamOld__pipe.wait_for_frames()
        aligned_frames = self._StreamOld__align_to_color.process(frames)
        rs_depth = aligned_frames.get_depth_frame()
        rs_color = aligned_frames.get_color_frame()

        np_depth = np.float32(rs_depth.get_data()) * 1 / 65535
        np_color = np.asanyarray(rs_color.get_data())

        depth = o3d.t.geometry.Image(o3d.core.Tensor(np_depth))
        color = o3d.t.geometry.Image(o3d.core.Tensor(np_color))
        self.frame = (depth, color)
        return self.frame

    def _get_config(self, device):
        """Set up configuration for the stream."""
        cfg = rs2.config()
        # Configure the camera with the default setting.
        cfg.enable_device(self.serial)

        # NEED TO TEST TO ENSURE L515 COMPATIBILITY
        cfg.enable_stream(rs2.stream.color, 1280, 720, rs2.format.rgb8, 30)  # Fails for L515 (possibly due to USB)
        cfg.enable_stream(rs2.stream.depth, 1280, 720, rs2.format.z16, 30)  # Fails for L515 (possibly due to USB)
        return cfg