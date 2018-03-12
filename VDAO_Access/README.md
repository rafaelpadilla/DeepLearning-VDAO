# **VDAO_Access Project** #

With **VDAO_Acess Project** you have access to different tools to:

* Play VDAO videos;
* Play VDAO videos highlighting lost objects with bounding boxes;
* Capture a specific frame within the videos;
* Split a video into different frames, skipping a given number of frames;
* Merge objects from [ALOI Database](http://aloi.science.uva.nl/) or **any database** with its corresponding mask;
* Merge objects from [ALOI Database](http://aloi.science.uva.nl/) or **any database** with any frame from VDAO videos;
* Given an object from [ALOI Database](http://aloi.science.uva.nl/) or **any available** object and its binary mask, you can obtain its exact bounding box.

Below you can see examples showing how to use some of these tools.

The examples shown below are also available in [this Demo script](Demo.py).

First, you need to add the following references into your code:

```python
import cv2
from VDAOVideo import VDAOVideo 
from VDAOHelper import VideoType
from ALOIHelper import ALOIDatabase
```

All manipulations to the frames will be made through the VDAOVideo object. In order to create the object, you need to pass to the constructor the path to a video from the VDAO database. You can download the videos from [here](https://github.com/rafaelpadilla/DeepLearning-VDAO/blob/master/VDAO.md).

**Example 00: Create the object VDAOVideo**
```python
# Passing the path and the enum that identifies if the video is a reference (it contains no lost objects) or if it is a target video (it contains lost objects):
myVideo = VDAOVideo("/home/rafael/Thesis/ref-mult-ext-part02-video01.avi", VideoType.Reference)

# If you don't rename the name of the file, you don't need to pass the enum. The API will identify if the video is a reference or a target one:
myVideo = VDAOVideo("/home/rafael/Thesis/ref-mult-ext-part02-video01.avi")
```

**Example 01: Obtain different information of a video**

You can have access to different information of your videos. Use the **get functions** to get individual information of the videos.

The code below shows how to obtain number of frames and frame rate of a video using the functions **getNumberOfFrames()** and **getFrameRate()**.

```python
# Get the number of frames
numberOfFrames = myVideo.videoInfo.getNumberOfFrames()
print("Number of frames: "+ numberOfFrames)
# Get frame rate
frameRate = myVideo.videoInfo.getFrameRate()
print("Frame rate: "+ frameRate)
```

Output: 
```python 
Number of frames: 29034  
Frame rate: 24/1
```

All the get functions are listed below:  

| # | Function | Description |
| :---: | :---: | :---: |
| 1 | ```getFilePath()``` | Gets full file path |
| 2 | ```getFileName()``` | Gets only the name of the file |
| 3 | ```getFormat()``` | Gets format of the file |
| 4 | ```getFormatLong()``` | Gets full format description |
| 5 | ```getSize()``` | Gets the size of the file in bytes |
| 6 | ```getCreationDate()``` | Gets the creation date and time of the file |
| 7 | ```getEnconderType()``` | Gets the encoder used to generate the file |
| 8 | ```getCodecType()``` | Gets the codec for the file |
| 9 | ```getCodecLongType()``` | Gets the full description of the codec |
| 10 | ```getWidth()``` | Gets the width (in pixels) of the frames |
| 11 | ```getHeight()``` | Gets the height (in pixels) of the frames |
| 12 | ```getWidthHeight()``` | Gets the width and height (in pixels) of the frames |
| 13 | ```getSampleAspectRatio()``` | Gets width x height ratio of the pixels with respect to the original source |
| 14 | ```getDisplayAspectRatio()``` | Gets width x height ratio of the data as it is supposed to be displayed |
| 15 | ```getPixelFormat()``` |  Gets the raw representation of the pixel |
| 16 | ```getFrameRate()``` | Gets number of frames that are displayed per second in the format X/1 |
| 17 | ```getFramesPerSecond()``` | Gets number of frames that are displayed per second |
| 18 | ```getDurationTs()``` | Gets the duration of the whole video in frames |
| 19 | ```getRealDuration()``` | Gets the full duration of the video in seconds |
| 20 | ```getBitRate()``` | Gets the number of bits used to represent each second of the video |
| 21 | ```getNumberOfFrames()``` | Gets the number of frames of the whole video |
  
**Example 02: Obtain all information of a video**

Besides number of frames and frame rate, you can use the function **printAllInformation()** to have the full set of information in the output window. See the example below:

```python
# Print all information about the video  
myVideo.videoInfo.printAllInformation()
```

Output: 
```python
*************************************
********** VDAO video info **********
 
File path: /home/rafael/Downloads/obj-mult-ext-part02-video01.avi
File name: obj-mult-ext-part02-video01.avi
File extension: avi (AVI (Audio Video Interleaved))
Created on: None
Encoder: MEncoder SVN-r35234-4.7.2
File size: 9125247280
Codec: h264 (H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10)
Width: 1280
Height: 720
Width x Height: [1280, 720]
Sample aspect ratio: 1:1
Display aspect ratio: 16:9
Pixel format: yuv420p
Frame rate: 24/1
Duration ts: 29034
Duration: 1209.750000 
```

**Example 03: Obtaining and saving specific frames within the video**

Use the function **getFrame(frameNumber, withInfo)** to obtain a specific frame within the video. You will just need to pass the frame number as the parameter. The parameter withInfo is a boolean where you should inform if you want your frame with or without informations.

The example below shows you how to skip 2 frames, going from frame 1 to 15 and save them:

```python
# Get every 2 frames starting from 1 to 14 and save them
# You can set the last parameter to True if you want to show details of 
# the frame, otherwise set it to False or don't pass anything.
myVideo.SkipAndSaveFrames(1, 14, 2, '/media/rafael/Databases', True)
```

You can set the last parameter to True if you want to show details of the frame, otherwise set it to False or don't pass anything. See the images below as examples:

<!--- Showing examples of frames with and without information--->
<div style="text-align:center">
<img src="https://github.com/rafaelpadilla/DeepLearning-VDAO/blob/master/VDAO_Access/images/ex_withWithoutInfo.jpg" alt="AAAAAA" style="width: 30px;"/>
<p align="center">Example of frames with info (showInfo=True) and without info (showInfo=False) respectively </p>
</div>


**Example 04: Play a VDAO video**

You can use the method **playVideo(showFrame, showBoundingBoxes)** specifying in the boolean parameter **showFrame** if you want to show the frame counter in the video or not.  

The code below shows how to play a VDAO video with and without frame information:

```python
# Play video showing information about the video
myVideo.PlayVideo(True)
# Play video showing information about the video
myVideo.PlayVideo(False)
```
With the parameter **showBoundingBoxes** set to ```True```, the video will be played and whenever there is a bounding box identified in the annotation file, the bounding box will be drawn in every frame. To use this feature, when creating the VDAOVideo object, it is necessary to pass the path for the annotation file using the parameter ```annotationFilePath```. See the example below:

```python
# Set the folder where the annotation file (.txt) and the video (.avi) are
folder = '/media/rafael/Databases/databases/VDAO/VDAO/Table_1-Shoe_Position_1'
# Set the paths for the video and its annotation file
video = os.path.join(folder, 'obj-sing-amb-part01-video01.avi')
annotation = os.path.join(folder, 'obj-sing-amb-part01-video01.txt')
# Create VDAOvideo object
vdao = VDAOVideo(video, annotationFilePath=annotation)
# Play the video setting the parameter showBoundingBoxes to True
vdao.PlayVideo(True, showBoundingBoxes=True)
```

The image below shows one of the frames of the example above where an object has its bounding box shown.

<!--- Showing examples of frames with and without information--->
<div style="text-align:center">
<img src="https://github.com/rafaelpadilla/DeepLearning-VDAO/blob/master/images/ex_withBBfromAnnotation.jpg" alt="AAAAAA" style="width: 30px;"/>
<p align="center">Example of PlayVideo function showing a bounding box obtained from the annotationFile (showBoundingBoxes=True).</p>
</div>


**Example 05: Crop an object given its mask**

If you want to obtain the exact bounding box of an object from your objects' database, you just have to call the static method **getBoundingBoxMask(mask)** of the class ObjectDatabase.  
The parameter **mask** is the loaded image containing the mask.

```python
# Read image from the ALOI database
mask = cv2.imread('/media/rafael/Databases/databases/ALOI/mask4/1/1_c1.png')
# Use static method to get the ROI of the mask
[min_x, min_y, max_x, max_y] = ObjectDatabase.getBoundingBoxMask(mask)
# Draw rectangle representing the ROI and show it
cv2.rectangle(mask,(min_x,min_y),(max_x,max_y),(0,255,0),1)
cv2.imshow('ROI', mask)
cv2.waitKey(100)
```

**Example 06: Merge an object with its mask**

Sometimes you will need to obtain random objects and crop them according to its mask. For that you have to instantiate the class ObjectDatabase and call its function **getRandomObject()**. See the example below.

```python
# Creating ObjectDatabase passing the images and masks paths
aloi = ObjectDatabase('/media/rafael/Databases/databases/ALOI/png4', \
                      '/media/rafael/Databases/databases/ALOI/mask4')
# Get a random object and its mask
[mergedImage, (min_x, min_y, max_x, max_y)] = aloi.getRandomObject()
cv2.rectangle(mergedImage,(min_x,min_y),(max_x,max_y),(0,255,0),1)
cv2.imshow('ROI', mergedImage)
cv2.waitKey(0)
```
The image below shows the application of the function ```getRandomObject```. The rectangle in the ouput image was added as demonstraded in the previous example.

<!--- Example of input and output of the function getRandomObject--->
<p align="center">
<img src="https://github.com/rafaelpadilla/DeepLearning-VDAO/blob/master/VDAO_Access/images/ex_mergedImages.jpg" alt="AAAAAA" style="width: 30px;"/>
<p align="center">Random image from the ALOI database was chosen and merged with its mask using the getRandomObject function. It results in the 3rd image (the bounding box was added afterwards)</p>
</div>
