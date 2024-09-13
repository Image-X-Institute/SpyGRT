# SpyGRT

SpyGRT, or Surface Guided Radiotherapy Toolkit in Python is a toolkit that allows fast development of novel surface guidance applications for radiotherapy. 

SpyGRT allows easy: 
- Operation of the depth camera (current model is Intel Realsense D415)
- Calibration of multiple cameras into the same external frame of reference
- Tracking of motion using an ICP workflow 

## Getting Started

### Installing SpyGRT using pip
The preferred method for installing SpyGRT is to use pip. This will download all the required packages, however a list of prerequisites and where to find them are listed in the next section in case you prefer to build them from source.

#### Steps to installing SpyGRT with pip

  1. Clone the SpyGRT GitHub repo
  2. Open a command prompt 
  3. Navigate to the SpyGRT folder (not the embedded spygrt folder)
  4. Enter the following pip command
  
```
pip install -e . 
```
  5. You should now be able to run SpyGRT

### Prerequesites

#### 1. Intel RealSense SDK2.0 

If you want to build from source, installation instructions available [here](https://github.com/IntelRealSense/librealsense). The code has currently only been tested up to release version 2.5.0 but should work with more recent releases. 

#### 2. Open3D library v.0.17.0 or later

If you want to build it from source, full release available [here](https://github.com/PointCloudLibrary/pcl/releases) and instructions to compile it yourself are available [here](http://www.open3d.org/docs/release/getting_started.html).
You will need to build from source to make use of the GPU

#### 3. OpenCV 

Full release available [here](https://opencv.org/releases/), if you prefer to build it from source. 

#### 4. Numpy
More Information available [here](https://numpy.org/install/)

### Working Examples
The example folder contain three examples that illustrate how to use the basic functions of SpyGRT. More instructions are located in that folder.

### Depth Sensor
SpyGRT is currently built to use [Intel RealSense D415](https://www.intelrealsense.com/depth-camera-d415/) stereo depth sensors for capturing depth data. However, it should be possible to modify the stream module to work with any commercial depth sensors that have a python API. 

### Intel RealSense Viewer
While SpyGRT contains the required code for real-time SGRT and analysis of prior recordings, it cannot currently be used to create new recordings. We suggest installing the free realsense-viewer app available [here](https://www.intelrealsense.com/sdk-2/) if you want to create your own recordings. 

## Calibrations Boards

### Default SpyGRT Calibration Board

SpyGRT uses a checkerboard to calibrate multiple cameras. By default, it assumes that the checkboard has 5 rows, 8 columns, and that the length of the sides of each square is 4.3 centimeters. If you want to print the default checkerboard, you can find it in a powerpoint file located in the SampleData folder. This file should be printed on an A3 paper to get the accurate dimensions. 

### Custom Calibration Board
Any checkerboard can be use to calibrate cameras with SpyGRT. If you want to use a custom checkerboard, you will need to change the following default variables in *calibration.py*

1. **DEFAULT_ROW**: Number of rows in the custom calibration board.
2. **DEFAULT_COLUMN**: Number of columns in the custom calibration board
3. **DEFAULT_SQUARE_SIZE**: Length in meters of the sides of a square in the custom calibration board 

## Tracking with SpyGRT
Calibration files are necessary to use the SpyGRT for tracking. By default, SpyGRT will look for calibration files *../Calibrations/*. If you want to change this path, you must change the following constant in *tracker.py*

- **CALIBRATION_FOLDER**

## Author
Youssef Ben Bouchta

Email: youssef.benbouchta@sydney.edu.au
