# Example of how to align two cameras with a real-time stream.
# Author: Youssef Ben Bouchta
# Last Edited: 29/03/2022

#Imports
import pyrealsense2 as rs2
from spygrt.calibration import Calibrator
from spygrt.stream import Camera

def main():
	
	#Initialisation of calibrator objects using real-time streams. Look at how the 
	# Recording example for how to use a Recording object instead.

	## Set up a realsense context to access connected devices
	ctx = rs2.context()

	## Initialising container for calibrator object
	calibs = []

	## For every realsense device connected, initialise a calibrator object
	for dev in ctx.query_devices():
		calibs.append(Calibrator(Camera(dev)))

	# Using the calibrator objects to obtain the calibration to the board frame of reference.

	## Align the frame of reference of the two cameras.
	calibs[0].align_cameras(calibs[1])

	## Align the camera frame of reference to the board frame of reference.
	calibs[1].align_to_board(cal2 = calibs[0])

	# Outputs calibration to a text file.
	for cal in calibs:
		cal.write_cal()
	
	print(calibs[0].T)
	print(calibs[1].T)

#So this method can be run on the terminal by calling python TrackingExample.py
if __name__ == "__main__":
	main()