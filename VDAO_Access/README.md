# **VDAO Access Project** #

1. [Intro](#intro_project)  
2. [Environment Preparation & Dependencies](#env_preparation)  
3. [Examples](#examples)  
   [- Example 00: Create the object VDAOVideo](#example00)  
   [- Example 01: Obtain different information of a video](#example01)    
   [- Example 02: Obtain all information of a video](#example02)    
   [- Example 03: Obtaining and saving specific frames within the video](#example03)    
   [- Example 04: Play a VDAO video](#example04)    
   [- Example 05: Get the bounding box of a mask](#example05)    
   [- Example 06: Merge an object with its mask](#example06)    
   [Example 07: Get random ALOI object and merge it with its mask)(#example07) 
4. [FAQ](#FAQ)

<a id="intro_project"></a>  
## Intro  

The **VDAO Access Project** provides you the following tools to handle VDAO videos:

* Play VDAO videos;
* Play VDAO videos highlighting lost objects with bounding boxes;
* Capture a specific frame within the videos;
* Split a video into different frames, skipping a desired number of frames;
* Merge objects from [ALOI Database](http://aloi.science.uva.nl/) or **any database** with its corresponding mask;
* Merge objects from [ALOI Database](http://aloi.science.uva.nl/) or **any database** with any frame from VDAO videos;
* Given an object from [ALOI Database](http://aloi.science.uva.nl/) or **any available** object and its binary mask, you can obtain its exact bounding box;

Below you can find examples showing how to use some of these tools.

<a id="env_preparation"></a>
## Environment Preparation & Dependencies  

The **VDAO Access** project was developed and tested using **Python version 3.6** in Linux environment.  

You need **openCV** and **ffmpeg** installed to run this project. We recommend you to use an environment manager such as Conda or virtualenvwrapper.

Use the commands below to create a **Conda environment** and install openCV and ffmpeg packages (if do not have Conda, install it following the instructions [here](https://conda.io/docs/user-guide/install/index.html)):

#### Creating a new environment:  

Type the command below in your prompt to create a new environment. Replace ```myenv``` with the environment name you want to create.  
```
conda create -n myenv python=3.6
```

A list of packages to be installed will be shown. Confirm the installation by typing **y**. 

#### Activating your environment:

Use the following command to activate your new environment:
```
source activate myenv
```

You will see ```(myenv)``` at the left side of the Unix prompt. It means your environment was created and is active.

#### Installing openCV:

Now you have your environment activated, type the command below to install openCV:
```
conda install -c menpo opencv3 
```
A list of packages to be installed (including opencv3) will be presented. Confirm with **y** to proceed. 

#### Installing ffmpeg:

Another dependency of the VDAO Project is the package ffmpeg. With your environment activate, type the command below to install it:

```
conda install -c menpo ffmpeg 
```
Confirm the installation by typing **y** 

With these packages installed, you need to clone or download the **VDAO Project** to start playing with VDAO. :) Have fun!

#### Importing headers:

Don't forget the headers! In order to run the examples or start using the VDAO Project functions, you need to add the following references into your code:

```python
import os
import cv2
from VDAOVideo import VDAOVideo 
from VDAOHelper import VideoType
from ObjectHelper import ObjectDatabase
```
<a id="examples"></a> 
# Examples  

<a id="example00"></a>
## Example 00: Create the object VDAOVideo  

All manipulations to the frames will be made through the ```VDAOVideo``` object. To do so, you need to pass to the constructor the path to a video from the VDAO database. (You can download the videos from [here](https://github.com/rafaelpadilla/DeepLearning-VDAO/blob/master/VDAO.md)).

Using a video path to create the ```VDAOVideo``` object:
```python
# Creating a VDAOVideo object using a video path.
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi")
```
You can also create the ```VDAOVideo``` object identifying your video as being a reference video or a video containing objects by using the enums ```VideoType.Reference``` or ```VideoType.WithObjects``` respectively.
```python
# Creating a VDAOVideo object identifying it as a reference video (it contains no lost objects)
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi", VideoType.Reference)
```
<a id="example01"></a>  
## Example 01: Obtain different information of a video  

You can have access to different information of your videos. Use the ```get functions```(#get_functions) to get specific information of the videos.

The code below shows how to obtain number of frames and frame rate of a video using the functions ```getNumberOfFrames()``` and ```getFrameRate()```.

```python
# Get the number of frames
numberOfFrames = myVideo.videoInfo.getNumberOfFrames()
print("Number of frames: "+ str(numberOfFrames))
# Get frame rate
frameRate = myVideo.videoInfo.getFrameRate()
print("Frame rate: "+ frameRate)
```

Output: 
```python 
Number of frames: 18969
Frame rate: 24/1
```

<a id="get_functions"></a>
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
| 17 | ```getDurationTs()``` | Gets the duration of the whole video in frames |
| 18 | ```getRealDuration()``` | Gets the full duration of the video in seconds |
| 19 | ```getBitRate()``` | Gets the number of bits used to represent each second of the video |
| 20 | ```getNumberOfFrames()``` | Gets the number of frames of the whole video |
  
<a id="example02"></a>
## Example 02: Obtain all information of a video  

Besides the number of frames and frame rate, you can use the function ```printAllInformation()``` to have the full set of information in the output window. See the example below:

```python
# Print all information about the video  
myVideo.videoInfo.printAllInformation()
```

Output: 
```python
*************************************
********** VDAO video info **********
 
File path: /home/user/VDAO/ref-mult-ext-part02-video01.avi
File name: ref-mult-ext-part02-video01.avi
File extension: avi (AVI (Audio Video Interleaved))
Created on: None
Encoder: MEncoder SVN-r35234-4.7.2
File size: 5861580028
Codec: h264 (H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10)
Width: 1280
Height: 720
Width x Height: [1280, 720]
Sample aspect ratio: 1:1
Display aspect ratio: 16:9
Pixel format: yuv420p
Frame rate: 24/1
Duration ts: 18969
Duration: 790.375000
Bit rate: 59328974
Number of frames: 18969
 
*************************************
```

<a id="example03"></a>  
## Example 03: Obtaining and saving specific frames within the video  

Use the function ```getFrame(frameNumber, withInfo``` to obtain a specific frame within the video. You just need to pass the frame number as the parameter. The parameter ```withInfo``` is a boolean informing if you want an output image with or without information.  

The example below shows you how to skip 2 frames, going from frame 1 to 15 and save them:

```python
# Get every 2 frames starting from 1 to 15 and save them
myVideo.SkipAndSaveFrames(startingFrame=1, endingFrame=15, framesToSkip=2, outputFolder='/home/user/VDAO/outputs', filePrefix='myFrame_', showInfo=True)
```
The output confirms that the frames were created:
```python
File sucessfully saved: /home/user/VDAO/outputFolder/myFrame_1.jpg
File sucessfully saved: /home/user/VDAO/outputFolder/myFrame_3.jpg
File sucessfully saved: /home/user/VDAO/outputFolder/myFrame_5.jpg
File sucessfully saved: /home/user/VDAO/outputFolder/myFrame_7.jpg
File sucessfully saved: /home/user/VDAO/outputFolder/myFrame_9.jpg
File sucessfully saved: /home/user/VDAO/outputFolder/myFrame_11.jpg
File sucessfully saved: /home/user/VDAO/outputFolder/myFrame_13.jpg
File sucessfully saved: /home/user/VDAO/outputFolder/myFrame_15.jpg
```

Use the parameters ```filePrefix``` to name your images starting with a desired word.  
Set the ```showInfo``` parameter to ```True``` if you want to show details of the frame in your output image, otherwise set it to ```False``` (default). See the images below as examples:  

<!--- Showing examples of frames with and without information--->
<div style="text-align:center">
<img src="https://github.com/rafaelpadilla/DeepLearning-VDAO/blob/master/VDAO_Access/images/ex_withWithoutInfo.jpg" alt="Image" style="width: 30px;"/>
<p align="center">Example of frames with info (showInfo=False) and without info (showInfo=True) respectively </p>
</div>

<a id="example04"></a>  
## Example 04: Play a VDAO video  

You can use the method ```playVideo(showInfo, showBoundingBoxes)``` passing ```True``` or ```False``` to the parameter ```showInfo``` idicating if you want to show the frame counter in the video or not.  

The code below shows how to play a VDAO video with and without frame information:

```python
# Play video showing information about the video
myVideo.PlayVideo(showInfo=True)
# Play video showing information about the video
myVideo.PlayVideo(showInfo=False)
```
With the parameter ```showBoundingBoxes``` set to ```True```, the video will be played and whenever there is a bounding box identified in the annotation file, the bounding box will be drawn in its respective frame. It is necessary to pass the path for the annotation file using the parameter ```annotationFilePath``` in the constructor. See the example below:

```python
# Set the folder where the annotation file (.txt) and the video (.avi) are
folder = '/home/user/VDAO'
# Set the paths for the video and its annotation file
videoPath = os.path.join(folder, 'obj-sing-amb-part01-video02.avi')
annotationPath = os.path.join(folder, 'obj-sing-amb-part01-video02.txt')
# Create VDAOvideo object
vdao = VDAOVideo(videoPath, annotationFilePath=annotationPath)
# Play the video setting the parameter showBoundingBoxes to True
vdao.PlayVideo(showBoundingBoxes=True)
```

The image below shows one of the frames of a VDAO video with the bounding boxes obtained from the annotation file.

<!--- Showing examples of frames with and without information--->
<div style="text-align:center">
<img src="https://github.com/rafaelpadilla/DeepLearning-VDAO/blob/master/images/ex_withBBfromAnnotation.jpg" alt="AAAAAA" style="width: 30px;"/>
<p align="center">Example of PlayVideo function showing a bounding box obtained from the annotationFile (showBoundingBoxes=True).</p>
</div>

<a id="example05"></a>  
## Example 05: Get the bounding box of a mask  

If you want to obtain the exact bounding box of a mask, you just have to call the static method ```getBoundingBoxMask(mask)``` of the class ```ObjectDatabase```.  
The parameter ```mask``` can be the path of the mask or a loaded mask as shown below.

```python
# Read image from the ALOI database
mask = cv2.imread('/home/user/ALOI/mask4/1/1_c1.png')
# Use static method to get the ROI of the mask
[min_x, min_y, max_x, max_y] = ObjectDatabase.getBoundingBoxMask(mask)
# Draw rectangle representing the ROI and show it
cv2.rectangle(mask,(min_x,min_y),(max_x,max_y),(0,255,0),1)
# Show image
cv2.imshow('ROI', mask)
cv2.waitKey(0)
```
<a id="example06"></a>  
## Example 06: Merge an object with its mask 

If you have an object and its mask, you can merge them by calling the static function ```ObjectDatabase.blendImageAndMask(objPath, maskPath)```. This function loads the images, applies the operation ```np.multiply(img,mask/255)``` and returns the resulting image.

```python
# Read image from the ALOI database
myObjectPath = '/home/user/ALOI/png4/259/259_c.png'
myMaskPath = '/home/user/ALOI/mask4/259/259_c.png'
blendedImage = ObjectDatabase.blendImageAndMask(objPath = myObjectPath, maskPath=myMaskPath)
cv2.imshow('Blended image', blendedImage)
cv2.waitKey(0)
```

<a id="example07"></a>  
## Example 07: Get random ALOI object and merge it with its mask 

Sometimes you will need to obtain random objects and merge them according to its mask. For that you have to instantiate the class ```ObjectDatabase``` and call its function ```getRandomObject()```. 
The function ```getRandomObject()``` gets a random object and finds its mask by matching the names. For example, the mask of the image _~/ALOI/png4/762/762_r.png_ needs to be _~/ALOI/mask4/762/762_r.png_. See the example below.

```python
# Creating ObjectDatabase passing the images and masks paths
aloi = ObjectDatabase(imagesPath='/home/user/ALOI/png4', \
                      masksPath='/home/user/ALOI/mask4')
# Get a random object and its mask
[mergedImage, (min_x, min_y, max_x, max_y)] = aloi.getRandomObject()
# Draw a green rectangle around the merged image
cv2.rectangle(mergedImage,(min_x,min_y),(max_x,max_y),(0,255,0),1)
cv2.imshow('image', mergedImage)
cv2.waitKey(0)
```
The image below shows the application of the function ```getRandomObject()```. The rectangle in the ouput image was added as demonstraded in the previous example.

<!--- Example of input and output of the function getRandomObject--->
<p align="center">
<img src="https://github.com/rafaelpadilla/DeepLearning-VDAO/blob/master/VDAO_Access/images/ex_mergedImages.jpg" alt="AAAAAA" style="width: 30px;"/>
<p align="center">Random image from the ALOI database was chosen and merged with its mask using the getRandomObject function. It results in the 3rd image (the bounding box was added afterwards)</p>
</div>

## FAQ <a id="FAQ"></a>  

**Q: How do I slice a video from frame 1 to 1001 skipping every 100 frames and save it in JPEG format?**
```python
# Create myVideo object
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi")
# Save videos skipping every 100 frames and save it in the output folder
# The jpegQuality can go from 0 to 100 (the higher, the better). Default is 95.
myVideo.SkipAndSaveFrames(startingFrame=1, endingFrame=1001, framesToSkip=100, \
outputFolder='/home/user/VDAO/outputs', \
extension=ImageExtension.JPG, jpegQuality=100)
```

**Q: How do I slice a video from frame 1 to 1001 skipping every 100 frames and save it in PNG format?**
```python
# Create myVideo object
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi")
# Save videos skipping every 100 frames and save it in the output folder
# The compressionLevel can go from 0 to 9 (higher value means a smaller size
# and longer compression time). Default is 3.
myVideo.SkipAndSaveFrames(startingFrame=1, endingFrame=1001, framesToSkip=100, \
outputFolder='/home/user/VDAO/outputs', \
extension=ImageExtension.PNG, compressionLevel=1)
```

**Q: How do I visualize a specific frame within a video?**
```python
# Create myVideo object
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi")
# Get frame number 530
ret, frame = myVideo.GetFrame(530)
# Check if frame was successfully retrieved 
if ret:
    # Show frame
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
```

**Q: How to know which compression method was used in a video?**
```python
# Create myVideo object
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi")
# Get codec info
codec = myVideo.videoInfo.getCodecLongType()
print('Codec: %s' % codec)
```

**Q: How do I get the frame rate of a  video?**
```python
# Create myVideo object
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi")
# Get frame rate
frameRate = myVideo.videoInfo.getFrameRate()
print('Frame rate: %s' % frameRate)
```

**Q: How do I get the total number of frame in a video?**
```python
# Create myVideo object
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi")
# Get frames
totalFrames = myVideo.videoInfo.getNumberOfFrames()
print('Total frames: %s' % totalFrames)
```

**Q: How to merge an image and its mask in a single image?**
```python
# Get image and mask paths
myObjectPath = '/home/user/ALOI/png4/259/259_c.png'
myMaskPath = '/home/user/ALOI/mask4/259/259_c.png'
# Merge image with its mask
blendedImage = ObjectDatabase.blendImageAndMask(objPath = myObjectPath, maskPath=myMaskPath)
# Show result
cv2.imshow('Blended image', blendedImage)
cv2.waitKey(0)
```

**Q: How to obtain a bounding box of a mask?**
```python
# Get mask path
myMaskPath = '/home/user/ALOI/mask4/259/259_c.png'
# Get bounding box
[min_x, min_y, max_x, max_y] = ObjectDatabase.getBoundingBoxMask(myMaskPath)
print("Upper left point: (%s, %s)" % (min_x, min_y))
print("Width: %s" % (max_x-min_x))
print("Height: %s" %  (max_y-min_y))
```


**Q: How do I play a VDAO video?**
```python
# Create myVideo object
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi")
# Play
myVideo.PlayVideo(showInfo=False)
```

**Q: How do I play a VDAO video showing the bounding boxes in the annotation file?**
```python
# Create myVideo object
myVideo = VDAOVideo(videoPath="/home/user/VDAO/obj-sing-amb-part01-video02.avi", annotationFilePath='/home/user/VDAO/obj-sing-amb-part01-video02.txt')
# Play
myVideo.PlayVideo(showInfo=False, showBoundingBoxes=True)
```

**Q: How do I add an object in an image making a smooth transition in the border regions?**
```python
objectPath = '/home/user/ALOI/png4/259/259_c.png'
maskPath = '/home/user/ALOI/mask4/259/259_c.png'
backgroundPath = '/home/user/Backgrounds/bg_1.jpg'
# Set some parameters for the image
scale = 1
angle = 30
# Blend mask, image and background
resultImage, _ = ObjectDatabase.blendImageAndBackground_2(objectPath, maskPath, backgroundPath, xIni=20, yIni=70, scaleFactor=scale, rotAngle=angle)
# Show result
cv2.imshow('final image', resultImage/255)
cv2.waitKey(0)
```
