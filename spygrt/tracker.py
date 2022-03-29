import cv2 as cv
import pyrealsense2 as rs2
import open3d as o3d
import numpy as np
from  spygrt.stream import NoMoreFrames
import os
import sys


# Modify with path to calibration files
CALIBRATION_FOLDER = '../Calibrations/'

class Tracker():
	"""Class to handle surface tracking"""

	def __init__(self, devices, cam_dict = None):
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
			print("WARNING: no input camera directory, camera position will be assigned postion based on input order from right to left")
			self.r_sensor = cam1
			self.l_sensor = cam2

		self.refresh_frames()
		self.r_o3d_intrinsics = self.r_sensor.get_o3d_intrinsics()
		self.l_o3d_intrinsics = self.l_sensor.get_o3d_intrinsics()

		self.r_T = self.load_cal(CALIBRATION_FOLDER + self.r_sensor.serial + "_cal.txt")
		self.l_T = self.load_cal(CALIBRATION_FOLDER + self.l_sensor.serial + "_cal.txt")

		#This should be replaced by an ROI selection tool.
		self.compute_pcd(u_limit = [0.30,0.30,0.30], l_limit = [-0.30,-0.30, 0.01])
		self.ref_pcd = self.pcd

	def refresh_frames(self):
		"""Fetch the next frame from each sensor and apply a bilateral filter to the depth data.
		
		Warnings: This function can raise an unchecked exception if stream has ended.
		"""
		self.r_depth, self.r_color = self.r_sensor.get_frames()
		self.l_depth, self.l_color = self.l_sensor.get_frames()

		self.r_depth = self.r_depth.filter_bilateral(10,20/65535,10).linear_transform(scale=65535/1000)
		self.l_depth = self.l_depth.filter_bilateral(10,20/65535,10).linear_transform(scale=65535/1000)
		self.r_frame = o3d.t.geometry.RGBDImage(self.r_color, self.r_depth, aligned = True)
		self.l_frame = o3d.t.geometry.RGBDImage(self.l_color, self.l_depth, aligned = True)

	def compute_pcd(self, limit = None, u_limit = None, l_limit = None):
		"""Compute the point cloud captured in the current frame."""

		I = o3d.core.Tensor(np.identity(4),o3d.core.Dtype.Float32)
		if u_limit is not None and l_limit is not None:
			l_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(self.l_frame,
														   o3d.core.Tensor(self.l_o3d_intrinsics.intrinsic_matrix,dtype = o3d.core.Dtype.Float32),
														   I,1, depth_max = 1.2, with_normals = True).transform(self.l_T).to_legacy()
			r_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(self.r_frame,
															o3d.core.Tensor(self.r_o3d_intrinsics.intrinsic_matrix,dtype = o3d.core.Dtype.Float32),
															I,1, depth_max = 1.2, with_normals = True).transform(self.r_T).to_legacy()
			box = o3d.geometry.AxisAlignedBoundingBox(l_limit, u_limit)
			r_pcd = r_pcd.crop(box)
			l_pcd = l_pcd.crop(box)
			r_pcd = o3d.t.geometry.PointCloud.from_legacy(r_pcd)
			l_pcd = o3d.t.geometry.PointCloud.from_legacy(l_pcd)
			self.pcd = r_pcd + l_pcd
			self.pcd = self.pcd.voxel_down_sample(voxel_size = 0.001)

		elif limit is not None:
			l_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(self.l_frame,
														   o3d.core.Tensor(self.l_o3d_intrinsics.intrinsic_matrix,dtype = o3d.core.Dtype.Float32),
														   I,1, depth_max = 1.2, with_normals = True).transform(self.l_T).to_legacy()
			r_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(self.r_frame,
															o3d.core.Tensor(self.r_o3d_intrinsics.intrinsic_matrix,dtype = o3d.core.Dtype.Float32),
															I,1, depth_max = 1.2, with_normals = True).transform(self.r_T).to_legacy()
			box = o3d.geometry.AxisAlignedBoundingBox(-1*limit, limit)
			r_pcd = r_pcd.crop(box)
			l_pcd = l_pcd.crop(box)
			r_pcd = o3d.t.geometry.PointCloud.from_legacy(r_pcd)
			l_pcd = o3d.t.geometry.PointCloud.from_legacy(l_pcd)
			self.pcd = r_pcd + l_pcd
			self.pcd = self.pcd.voxel_down_sample(voxel_size = 0.001)

		else:
			l_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(self.l_frame,
														   o3d.core.Tensor(self.l_o3d_intrinsics.intrinsic_matrix,dtype = o3d.core.Dtype.Float32),
														   I,1, depth_max = 1.2, with_normals = True).transform(self.l_T)
			r_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(self.r_frame,
															o3d.core.Tensor(self.r_o3d_intrinsics.intrinsic_matrix,dtype = o3d.core.Dtype.Float32),
															I,1, depth_max = 1.2, with_normals = True).transform(self.r_T)
			self.pcd = r_pcd + l_pcd
			self.pcd = self.pcd.voxel_down_sample(voxel_size = 0.001)

	def reinitialize(self, limit = None, u_limit = None, l_limit = None):
		"""Reset the reference pointcloud to the current view.
		
		Warnings: This function can raise an unchecked exception if stream has ended.
		"""
		self.refresh_frames()
		if limit is not None:
			self.ref_pcd = self.compute_pcd(limit = limit)
		else:
			self.ref_pcd = self.compute_pcd(u_limit = u_limit, l_limit = l_limit)
	
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

		#Initializing Multiscale parameters
		#For downsampling of the pointcloud
		voxel_radius = [0.05,0.01,0.005,0.003,0.001,0.0005]
		#Maximum distance between correspondances
		dist = [0.1, 0.05, 0.01, 0.008, 0.005,0.001]
		#Number of iteration for the RMSE optimization
		max_iter = [50, 30, 14, 7,3]

		#Initial transformation guess
		T = np.identity(4)

		#Initializing pointclouds
		target = self.ref_pcd
		source = self.pcd

		for scale in range(5):
			iter = max_iter[scale]
			radius = voxel_radius[scale]
			max_distance = dist[scale]

   			#Downsample pointclouds
			source_d = source.voxel_down_sample(radius).to_legacy()
			target_d = target.voxel_down_sample(radius).to_legacy()

			#Generalized ICP

			result_icp = o3d.pipelines.registration.registration_generalized_icp(source_d, target_d,max_distance, T,
				o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
				o3d.pipelines.registration.ICPConvergenceCriteria(
					relative_fitness=1e-9,
					relative_rmse=1e-9,
					max_iteration=iter),
				)
			T = result_icp.transformation

		return o3d.core.Tensor(T,dtype = o3d.core.Dtype.Float32)

	def ref_track(self, length, filename, limit = None, u_limit = None, l_limit = None, verbose = True):
		"""
		Compute the motion between the current view and reference view for a certain duration. The duration is 
		given as input in units of # of frames. Input limits parameter can be used to crop the background.
		Verbose input option will enable progress tracking through printing on the console.
		Refer to ref_motion method for tracking details. 

		Output is in the form of a text file with the input filename. If the filename doesn't include a complete path,
		it will be saved in the current working directory. Filename should end with the file extension ".txt".
		"""

		#Initialize output storage array
		data = []
		for i in range(length):
			try:
				self.refresh_frames()
			except NoMoreFrames:
				print("Stopping loop early as we reached the end of the stream at frame: " + str(i))
				break
			self.compute_pcd(limit = limit, u_limit = u_limit, l_limit = l_limit)
			T = self.ref_motion()
			data.append(np.concatenate((Tracker.compute_translation(T),Tracker.compute_rotation(T))))
			if verbose:
				if i%5 == 0:
					print("Finished computing registration for frame:" + str(i))

		np.savetxt(filename, data)
	
	def load_cal(self, filename):
		T = np.loadtxt(filename)
		return o3d.core.Tensor(T, dtype = o3d.core.Dtype.Float32)

	def new_stream(self, devices, cam_dict = None):
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
			print("WARNING: no input camera directory, camera position will be assigned postion based on input order from right to left")
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
		return (np.linalg.norm(np.identity(3,dtype=R.dtype)-np.dot(R.T, R)) < 1e-6)

	@staticmethod
	def compute_translation(T):
		"""Extract translations from affine tranformation matrix."""
		return T.numpy().T[-1][0:3]

	@staticmethod
	def compute_rotation(T):
		"""Extract rotations from an affine transformation matrix."""
		R = T.numpy()[0:3].T[0:3].T.copy()
		if not Tracker.isR_matrix(R):
			print("Matrix is not a rotation matrix", sys.stderr)
			sys.exit()

		sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
		singular = sy<1e-6

		if not singular:
			roll = np.arctan2(R[2,1] , R[2,2]) * 180/np.pi
			pitch = np.arctan2(-R[2,0], sy) * 180/np.pi
			yaw = np.arctan2(R[1,0], R[0,0]) * 180/np.pi


		else :
			roll = np.arctan2(-R[1,2], R[1,1])
			pitch = np.arctan2(-R[2,0], sy)
			yaw = 0

		return [yaw, pitch, roll]

