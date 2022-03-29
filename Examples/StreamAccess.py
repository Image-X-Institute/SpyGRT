# Example of how to initialise stream objects and basic stram control commands.
# Author: Youssef Ben Bouchta
# Last Edited: 29/03/2022

#Imports
import pyrealsense2 as rs2
import open3d as o3d
import numpy as np
from spygrt.stream import Recording, Camera, NoMoreFrames

def main():
	# Initialisation of a Recording

    ## Filename of the recording
    ## Example filename: 'C:/Data/recording.bag'
    ## NB: Backslashes( \ ) do not work for file path on windows
    ## NB: Complete filepath can be ommited for files located in the working directory
    ## NB: Filepath relative to current working directory are also allowed.

    filename = '../SampleData/Cam1_Obj.bag'

    myRecording = Recording(filename)

    #Initialisation of a real-time stream, i.e: a Camera object

    ctx = rs2.context()
    ## Obtaining the first camera in the list of all realsense devices.
    ## NB: If multiple realsense devices are connected, the first will be random. 
    myFirstRealSenseDevice = ctx.query_devices()[0]
    myCamera = Camera(myFirstRealSenseDevice)

    # This part of this example will work for all stream objects.

    # If you want to work with a real-time stream, replace myRecording with myCamera.
    myStream = myRecording 

    # Accessing device informations

    # Print serial number of the camera that acquire(s/d) the stream
    print("Serial Number: " + str(myStream.serial))

    # Print model of the camera that acquire(s/d) the stream
    print("Camera Model: " + str(myStream.model))

    # Print the depth scale -> depth data * depth scale = depth in meters
    print("Depth Scale: " + str(myStream.depthscale))

    # Print the depth camera pinhole camera intrinsics - Needed for transforming depth frames to point clouds.
    # NB: the intrinsics will be given as an open3D intrinsics object.
    intrinsics = myStream.get_o3d_intrinsics()
    print("Camera intrinsics: \n" + str(intrinsics.intrinsic_matrix))

    ## Accessing frame acquired during initialisation.
    ## The frame parameter can also be used to access the last acquired frame.
    old_frame = myStream.frame

    # Acquire and access a new frame
    # NB: get_frames can raise an unhandled exception if it is not possible to acquire a new frame
    # This can occur if you've reached the end of a recorded stream or if the end_stream method has 
    # been called.
    try:
        new_frame = myStream.get_frames()
    except NoMoreFrames:
        print("Encountered a NoMoreFrames error")
        pass

    # NB: The first frame of a stream can be accessed with the first call to get_frames() or with the frame parameter

    # Obtaining an open3D pointcloud with depth data in meters


    #End the streams

    myStream.end_stream()

    # Accessing the depth frame and color frame separately

    depth_frame = new_frame[0]
    color_frame = new_frame[1]

    # Converting to Open3D tensors

    depth_tensor = depth_frame.as_tensor()
    color_tensor = color_frame.as_tensor()

    # Converting to numpy array

    depth_array = depth_tensor.numpy()
    color_array = color_tensor.numpy()

    # Extrinsic matrix - replace with actual matrix if known
    # NB: Extrinsic matrix represent the transformation from lab to camera, calibrator objects give the transformation matrix from camera to lab,
    ### these two matrices are inverses.
    ext = o3d.core.Tensor(np.identity(4),o3d.core.Dtype.Float32)

    ### Create an RGB+Depth image
    rgbd = o3d.t.geometry.RGBDImage(color_frame, depth_frame, aligned = True)

    ### Create point cloud with maximum depth of 2 meters
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, 
                                                           o3d.core.Tensor(intrinsics.intrinsic_matrix,
                                                                           dtype = o3d.core.Dtype.Float32),
                                                           ext, 1000/65535, depth_max = 2)

    # Extracting points position information as an Open3D tensor

    pcd_xyz_tensor = pcd.point['positions']

    # Extracting points color information as an Open3D tensor

    pcd_RGB_tensor = pcd.point['colors']

    # converting both to numpy array

    pcd_xyz_numpy = pcd_xyz_tensor.numpy()
    pcd_RGB_numpy = pcd_RGB_tensor.numpy()

    #Concatenating into a single Nx6 numpy array (x,y,z,R,G,B)

    pcd_numpy = np.concatenate((pcd_xyz_numpy, pcd_RGB_numpy),1)


#So this method can be run on the terminal by calling python TrackingExample.py
if __name__ == "__main__":
	main()