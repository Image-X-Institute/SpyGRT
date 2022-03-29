# Author: Youssef Ben Bouchta
# Last Edited: 20/03/2022
# Example showing how to use SpyGRT for tracking on pre-recorded images with precalibrated cameras.

#Imports
from spygrt.tracker import Tracker
from spygrt.stream import Recording

def main():
	"""Main loop doing the initialisation and tracking."""

	# filenames of the recordings
	# Example filename: 'C:/Data/recording.bag'
	# NB: Backslashes( \ ) do not work for file path on windows
	# NB: Complete filepath can be ommited for files located in the working directory
	# NB: Edit with the name of the files you want to use for tracking.
	filenames = ('../SampleData/Cam1_Obj.bag', '../SampleData/Cam2_Obj.bag')

	# Initialising streamable object

	devices = (Recording(filenames[0]), Recording(filenames[1]))

	# Initialising tracker object

	## NB: the initialisation method assumes the existence of a file called rec.serial
	tracker = Tracker(devices)

	#Setting up the reference point cloud

	## NB: By default, the reference point cloud is set up as the entire point cloud from a frame acquired during 
	## the initialisation of the tracker object.

	## Upper and lower limit of a bounding box that serves to define your ROI. If left blank, the entire pointcloud will be used
	## as your reference point cloud. 
	u_limit = [0.15,0.15,0.30]
	l_limit = [-0.15,-0.30, 0.01]

	## Compute a bounded pointcloud of the frame acquired during the initialisation of the tracker object.
	tracker.compute_pcd(u_limit = u_limit, l_limit = l_limit)

	## Updating the refrence point cloud
	tracker.set_ref_pcd(tracker.pcd)

	# Intialising Parameters for the tracking method

	## Maximum number of frame to track
	max_f = 500

	## Output filepath to write the tracking data to
	datafile = 'data.txt'

	## Do you want a progress update printed on screen, True by default
	verbose = True

	# Starting the tracking
	tracker.ref_track(max_f, datafile, u_limit = u_limit, l_limit = l_limit, verbose = verbose)

#So this method can be run on the terminal by calling python TrackingExample.py
if __name__ == "__main__":
	main()