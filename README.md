# Detecting lost objects with Deep Learning

In this repository you can find my experiments using Deep Learning on VDAO database.

## VDAO Database ##

VDAO is a video database containing annotated videos in a cluttered industrial environment. The videos were captured using a camera on a moving platform.

The complete database comprises a total 6 multi-object, 56 single-object and 4 no-object (for reference purposes) footages, acquired with two different cameras and two different light conditions, yielding an approximate total of 8.2 hours of video.

See [here](http://www02.smt.ufrj.br/~tvdigital/database/objects/docs/an_annotated_video_database_for_abandoned_object_detection_in_a_cluttered_environment.pdf) the paper presenting the database. [Here](http://www02.smt.ufrj.br/~tvdigital/database/objects/page_01.html) you can have access to the database and its annotation file.

<!--- Showing examples of frames --->
<div style="text-align:center">
<img src="https://github.com/rafaelpadilla/Detecting-lost-objects-with-Deep-Learning/blob/master/images/ex_frames_reference.jpg" alt="AAAAAA" style="width: 30px;"/>
<p align="center">Examples of the VDAO dataset reference frames (no objects) </p>
</div>

<div style="text-align:center">
<img src="https://github.com/rafaelpadilla/Detecting-lost-objects-with-Deep-Learning/blob/master/images/ex_frames_target.jpg" style="width: 30px;"/>
<p align="center">Examples of the VDAO dataset target frames (objects manually annotated with bounding boxes)</p>
</div>

## YOLO ##

Yolo (You Only <del>Live</del> Look Once) is a real-time object detection and classification that obtained excellent results on the [Pascal VOC dataset](http://host.robots.ox.ac.uk:8080/pascal/VOC/). So far, yolo has two versions: **Yolo V1** and Yolo V2, also refered as **Yolo 9000**. Click on the image below to watch Yolo 9000's promo video.

<!--- Yolo's link for you tube --->
<p align="center">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=VOC3huqHrss"><img src="https://github.com/rafaelpadilla/Detecting-lost-objects-with-Deep-Learning/blob/master/images/yolo_youtube.jpg" width="427" height="240" align="center"/></a>
</p>

The authors have created a website explaining how it works, how to use it and how to train yolo with your images. Check the references below: 

YOLO: **You Only Look Once: Unified, Readl-Time Object Detection** (2016)  
(Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi)  
	[[site](https://arxiv.org/abs/1506.02640)] 
	[[pdf](https://bitbucket.org/rafaelpadilla/mythesis/src/ad0d4d320df4c5897bdda58bbffd83055902d98b/materials/%5Bpaper%5D%20YOLO.pdf)] 
	[[slides](https://bitbucket.org/rafaelpadilla/mythesis/src/ad0d4d320df4c5897bdda58bbffd83055902d98b/materials/%5Bslides%5D%20YOLO%20CVPR%202016.pdf)] 
	[[talk](https://www.youtube.com/watch?v=NM6lrxy0bxs)] 
	[[ted talk](https://www.youtube.com/watch?v=Cgxsv1riJhI)] 
	
**YOLO9000: Better, Faster, Stronger** (2017)  
(Joseph Redmon, Ali Farhadi)  
	[[site](https://arxiv.org/abs/1612.08242)] 
	[[pdf](https://bitbucket.org/rafaelpadilla/mythesis/src/636e8f075be4e5186777c66ddbe8cb2ad0797fab/materials/%5Bpaper%5D%20YOLO9000.pdf)] 
	[[talk](https://www.youtube.com/watch?v=GBu2jofRJtk)] 
	[[slides](https://bitbucket.org/rafaelpadilla/mythesis/src/ad0d4d320df4c5897bdda58bbffd83055902d98b/materials/%5Bslides%5D%20YOLO9000%20CVPR%202017.pdf)] 
	
**YOLO: People talking about it**  
	[[Andrew NG](https://www.youtube.com/watch?v=9s_FpMpdYW8)] 
	[[Siraj Raval](https://www.youtube.com/watch?v=4eIBisqx9_g)] 

**YOLO: People writing about it (Explanations and codes)**  
	[[Towards data science](https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006)]: A brief summary about yolo and how it works.  
	[[Machine Think blog](http://machinethink.net/blog/object-detection-with-yolo/)]: A brief summary about yolo and how it works.  
	[[Timebutt's github](https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/)]: A tutorial explaing how to train yolo 9000 to detect a single class object.  
	[[Timebutt's github](https://timebutt.github.io/static/understanding-yolov2-training-output/)]: Read this if you want to understand yolo's training output.  
	[[Cvjena's github](https://github.com/cvjena/darknet/blob/master/cfg/yolo.cfg)]: Comments of some of the tags used in the cfg files.  
	[[Guanghan Ning's blog](http://guanghan.info/blog/en/my-works/train-yolo/)]: A tutorial explaining how to train yolo v1 with your own data. The author used two classes (yield and stop signs).  
	[[AlexeyAB's github](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)]: Very good project forked from yolo 9000 supporting Windows and Linux.  
	[[Google's Group](https://groups.google.com/forum/#!forum/darknet)]: Excellent source of information. People ask and answer doubts about darknet and yolo.
