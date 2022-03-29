# Introduction to using SpyGRT

SpyGRT is meant to be a tool that can be built upon to create custom SGRT applications for research purposes. As such, there is no user interface and minimal executable code. The examples in this directory should serve as demonstration of how to the various functionalities of SpyGRT. 

Some of the examples are in the form of a Jupyter notebook. To open the notebook, you will need to download Jupyter Notebook, which you can do using: 

``` pip install notebook```

The Jupyter Notebook examples are also available in non-executable .py files for ease of access.


## StreamAccess
Shows how to use the two stream classes. 

## NewCalibration
Shows how to calibrate multiple cameras using a calibration board.

## TrackingExample
Simple program that loads a recording and finds the motion of an object from the first frame.

### Running TrackingExample
To run the tracking example, you will need to provide the calibrations of the cameras in the form of 4x4 matrix in a .txt file named: *Path_to_File/XXX_cal.txt* where XXX is the camera serial number. You will also need to change the constant **CALIBRATION_FOLDER** in *tracker.py* with the path to the calibration files. 

If you want to run the tracking example with the sample streams in the SampleData folder, the calibrations files are in the CalibrationResults subfolder. Assuming your working directory is the source directory, **CALIBRATION_FOLDER** should be changed to './SampleData/CalibrationResults/'. You can also use the \*.Calib.bag files in the SampleData directory to create your own calibration files. 

