# README #

Face detection and recognition project. Assigned at the UniFi, Faculty of Engineering, as the project related to the **Image and Video Analysis** exam, 17/18.  
Autors : Giovanni Bindi, Giuliano Gambacorta.

## Structure ##

+	`config.ini` : configuration file, required by [configparser](https://docs.python.org/3/library/configparser.html) contains these params:
	*	__confidence__ : float representing the confidence level for a face to be recognized. A face with a confidence score less than this parameter will not be considered for recognition (**default 8.**)
	*	__threshold__ : float representing the likelihood that two faces belong to the same person. Calculated as the [cosine distance](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.cosine.html) between two output vectors from the CNN. (**default 0.35**)
	*	__vgg_path__ : relative (or absolute) path to the pre-trained CNN model (default **None**)
	*	__haar_path__ : relative (or absolute) (default **cwd**, cloned within this repository)
*	`arch.py` : CNN architecture ([VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/), taken from this [source](https://aboveintelligent.com/face-recognition-with-keras-and-opencv-2baf2a83b799))
*	`utils.py` : helper functions needed for config parsing and management, timing and handling [pickle](https://docs.python.org/3/library/pickle.html) files used to store previously tagged identities.
*	`identificator.py` : class defining the Identificator object, whom is delegated to detect and recognize the faces seen in the video.
*	`main.py`  

## Requirements ##

*	Python (>= 3.5)
*	Tensorflow
*	Keras
*	OpenCV
*	SciPy
*	NumPy
*	Pickle
*	Configparser

### Files needed :

*	__vgg_face.mat__ : (MATLAB) weights for a 16 layers deep CNN developed and trained by the Visual Geometry Group, as described in this [paper](https://arxiv.org/pdf/1409.1556.pdf). Download from [here](http://www.vlfeat.org/matconvnet/models/vgg-face.mat) or follow the instructions in the next section. 
*	__haarcascade_frontalface_default.xml__ : pre-trained (frontal) face detection feature-based classifier, stored as an XML file.


__TODO__ : generate requirements.txt file for conda/pip

## How to run ##

*	clone this repo : `git clone https://github.com/w00zie/face_recog`
*	run `main.py`
*	once asked for config profile, select `0 - DEFAULT`, if any file needed for the execution is missing it'll be downloaded

Two strings are shown on top of every frame :  

*	Seen {} different people -> Number of different people seen in this and all the previous sessions (if any). Actual number of different individuals that interacted with the camera.
*	Last seen/recognized : {} -> If there is no previous knowledge provided this string represents the last different person who've been recognized (new person 0, new person 1...), besides it displays the name of the previously-tagged person recognized.


# DEMO #

*	First we ran `main.py` with our default configuration. No previous knowledge provided.  


![picture](images/first_prova.gif)

*	Once 3 new people were detected, we were shown their faces...  
![picture](images/gamba.jpg) ![picture](images/giova.jpg) ![picture](images/pala.jpg)  

... and we've been asked to insert their relative names to tag these newly detected identities.  


![picture](images/keyboard_input.png)

*	On the next execution the previously labelled people are now recognized and tagged.  


![picture](images/last_prova.gif)


