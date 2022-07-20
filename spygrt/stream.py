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

import logging
import pyrealsense2 as rs2
import open3d as o3d
import numpy as np
import time
from abc import ABC, abstractmethod


# Setting up a logger that will write log messages in the 'stream.log' file.
logger = logging.getLogger(__name__)
fh = logging.FileHandler('stream.log', 'w', 'utf-8')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('- %(module)s - %(levelname)-8s: %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


class NoMoreFrames(Exception):
    pass


class Stream(ABC):
    """Abstract Parent Class to handle the interactions with the RealSense API"""

    def __init__(self):
        """Initialise stream and define stream settings."""

    @abstractmethod
    def get_frames(self):
        pass

    @property
    @abstractmethod
    def frame(self):
        pass

    @frame.setter
    @abstractmethod
    def frame(self, new_frame):
        """
        Ensures that the frame attribute is only modified by fetching a new frame from the camera.

        Args:
            new_frame: New frame to replace existing frame.
        """
        logging.warning('Manually setting the \'frame\' attribute is not allowed.')

    @abstractmethod
    def end_stream(self):
        pass

    @abstractmethod
    def start_stream(self):
        pass


class Camera(Stream):
    """Class to handle the interaction with the Realsense API for real-time frame capture"""

    def __init__(self, device):
        """
        Initialise the camera settings and start the stream.

        Args:
            device: Realsense device object pointing to a camera(rs2.device).
        """
        self._frame = None

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
        self.depthScale = depth_sensor.get_depth_scale()

        # Pinhole camera intrinsics for depth camera
        self.intrinsics = depth_sensor.get_stream_profiles()[0].as_video_stream_profile().intrinsics

        if depth_sensor.supports(rs2.option.emitter_enabled):
            depth_sensor.set_option(rs2.option.emitter_enabled, True)

        if depth_sensor.supports(rs2.option.laser_power):
            laser = depth_sensor.get_option_range(rs2.option.laser_power)
            depth_sensor.set_option(rs2.option.laser_power, laser.max)

        try:
            depth_sensor.set_option(rs2.option.max_distance, 200)
        except RuntimeError:
            print("no maximum distance setting for camera model: " + self.model)

        try:
            depth_sensor.set_option(rs2.option.min_distance, 0)
        except RuntimeError:
            print("no minimum distance setting for camera model: " + self.model)

            # Warmup
            self.warmup()

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

    # Encoding will be deprecated later, for now used for backward compatibility
    def get_frames(self, encoding='o3d'):
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
        frames = self._pipe.wait_for_frames()
        aligned_frames = self.__align_to_color.process(frames)
        rs_depth = aligned_frames.get_depth_frame()
        rs_color = aligned_frames.get_color_frame()

        if encoding is 'o3d':
            np_depth = np.float32(rs_depth.get_data()) * 1 / 65535
            np_color = np.asanyarray(rs_color.get_data())

            depth = o3d.t.geometry.Image(o3d.core.Tensor(np_depth))
            color = o3d.t.geometry.Image(o3d.core.Tensor(np_color))
            self._frame = (depth, color)

        else:
            self._frame = (rs_depth, rs_color)
        return self._frame

    def warmup(self):
        """"Warmup the camera before frame acquisition"""
        frames = self._pipe.wait_for_frames()
        for i in range(29):
            frames = self._pipe.wait_for_frames()
        self.warmed = True
        self.intrinsics = frames.get_depth_frame().get_profile().as_video_stream_profile().intrinsics

    # Will be deprecated for using the intrinsics parameter.
    def o3d_intrinsics(self):
        """
        Transform the camera intrinsics to an open_3D format.

         Returns:
            intrinsics: Return the intrinsics of the camera in an open3D format.
        """
        return o3d.camera.PinholeCameraIntrinsic(self.intrinsics.width, self.intrinsics.height, self.intrinsics.fx,
                                                 self.intrinsics.fy, self.intrinsics.ppx, self.intrinsics.ppy)

    def end_stream(self):
        """End the stream."""
        self._pipe.stop()
        self.warmed = False

    def start_stream(self, rec=False, filename=None):
        """
        Starts or restarts the stream.

        Args:
            rec: Bool to indicate whether a recording should be created
            filename: filepath to the desired location for the recorded file.

        """
        if rec is True:
            if filename is None:
                filename = self.serial + time.localtime() + '.bag'
            else:
                try:
                    if filename[-4:] != '.bag':
                        filename = filename + '.bag'
                except TypeError:
                    logging.warning('filename is not a string, using default filename instead')
                    filename = self.serial + time.localtime() + '.bag'
            self._cfg.enable_record_to_file(filename)

        self._pipe.start(self._cfg)


class Recording(Stream):
    """Class to handle the interaction with the Realsense API for realsense bag files."""

    def __init__(self, device):
        """

        Args:
            device: link to a bag file containing a recording.
        """
        self._frame = None

        # pipeline that controls the acquisition of frames.
        self._pipe = rs2.pipeline()

        # Initialise config object
        self._cfg = rs2.config()

        # Configure the camera with the default setting.
        self._cfg.enable_device_from_file(device, False)

        # Start the pipeline - Frame fetching is possible after this.
        pipe_profile = self._pipe.start(self._cfg)

        # Object that align depth and color.
        self._align_to_color = rs2.align(rs2.stream.color)

        # Device object to access metadata.
        self._device = pipe_profile.get_device()

        # Serial number of the camera that captured this stream.
        self.serial = self._device.get_info(rs2.camera_info.serial_number)

        # Model of the camera that captured this stream.
        self.model = self._device.get_info(rs2.camera_info.name)

        # Access unit in which the depth is saved
        depth_sensor = self._device.first_depth_sensor()

        self.depthScale = depth_sensor.get_depth_scale()

        # Pinhole camera intrinsics for depth camera
        self.intrinsics = depth_sensor.get_stream_profiles()[0].as_video_stream_profile().intrinsics

        # Specify that this device is from a captured stream
        self._playback = self._device.as_playback()

        # Enable frame by frame access
        self._playback.set_real_time(False)

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

    def get_frames(self, encoding='o3d'):
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
                raise NoMoreFrames("Reached the end of the stream")
        except AttributeError:
            pass

        frames = self._pipe.wait_for_frames()
        aligned_frames = self._align_to_color.process(frames)
        rs_depth = aligned_frames.get_depth_frame()
        rs_color = aligned_frames.get_color_frame()

        if encoding is 'o3d':
            np_depth = np.float32(rs_depth.get_data()) * 1 / 65535
            np_color = np.asanyarray(rs_color.get_data())

            depth = o3d.t.geometry.Image(o3d.core.Tensor(np_depth))
            color = o3d.t.geometry.Image(o3d.core.Tensor(np_color))
            self._frame = (depth, color)
        else:
            self._frame = (rs_depth, rs_color)

        return self._frame

    def o3d_intrinsics(self):
        """
        Transform the camera intrinsics to an open_3D format.

        Returns:
            intrinsics: Return the intrinsics of the camera in an open3D format.
        """
        return o3d.camera.PinholeCameraIntrinsic(self.intrinsics.width, self.intrinsics.height, self.intrinsics.fx,
                                                 self.intrinsics.fy, self.intrinsics.ppx, self.intrinsics.ppy)

    def end_stream(self):
        """End the stream."""
        self._pipe.stop()

    def start_stream(self):
        """Starts or restarts the stream."""

        if self._playback.current_status() != rs2.playback_status.stopped:
            logging.warning('Tried to start a stream that is already started')
            return

        self._pipe.start(self._cfg)




class Stream_old:
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


class Recording_old(Stream_old):
    """Class to handle the interaction with the Realsense API for realsense bag files"""

    def __init__(self, bag_file):
        """Initialise playback setting and start the stream"""

        Stream_old.__init__(self, bag_file)

        # Specify that this device is from a captured stream
        self.__playback = self._Stream_old__device.as_playback()

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
            if not self.__playback.current_status() == rs2.playback_status.stopped and self._Stream_old__first:
                self._Stream__first = False
                return self.frame
            elif self.__playback.current_status() == rs2.playback_status.stopped:
                raise NoMoreFrames("Reached the end of the stream")
        except AttributeError:
            pass

        frames = self._Stream_old_pipe.wait_for_frames()
        aligned_frames = self._Stream_old__align_to_color.process(frames)
        rs_depth = aligned_frames.get_depth_frame()
        rs_color = aligned_frames.get_color_frame()

        if self._Stream__first:
            self.intrinsics = rs_depth.get_profile().as_video_stream_profile().intrinsics

        np_depth = np.float32(rs_depth.get_data()) * 1 / 65535
        np_color = np.asanyarray(rs_color.get_data())

        depth = o3d.t.geometry.Image(o3d.core.Tensor(np_depth))
        color = o3d.t.geometry.Image(o3d.core.Tensor(np_color))

        self.frame = (depth, color)
        return self.frame


class Camera(Stream_old):
    """Class to handle the interaction with the Realsense API for real-time frame capture"""

    def __init__(self, device):
        """Initialise the camera settings and start the stream"""

        # Initialise serial number early as it is needed for Stream configuration
        self.serial = device.get_info(rs2.camera_info.serial_number)

        # Indicate whether the camera has warmed up. If false, need to leave the camera running
        # for 10-15frames before any get_frames can be used.
        self.warmedup = False

        Stream_old.__init__(self, device)

        depth_sensor = self._Stream_old__device.first_depth_sensor()

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
            frames = self._Stream_old__pipe.wait_for_frames()
        self.warmedup = True
        self._Stream__first = False
        self.intrinsics = frames.get_depth_frame().get_profile().as_video_stream_profile().intrinsics

    def get_frames(self):
        """Fetch an aligned depth and color frame"""

        if not self.warmedup:
            self.warmup()
        frames = self._Stream_old__pipe.wait_for_frames()
        aligned_frames = self._Stream_old__align_to_color.process(frames)
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
