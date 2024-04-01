# YoloV3 custom dataset
In this project we present a model to train Yolo V3 for a custom dataset (Macroinvertebrates). In this version we address only a sample of Class Elmis_anea_adul. Also, we used the features detected in the morphotaxonomic concordance evaluation as classes to detect. 

## Original paper:
https://pjreddie.com/media/files/papers/YOLOv3.pdf

# Implementation:

### Prepare Dataset
  Execute labelImg.py
  
  	$ cd labelImg
  
  	$ python labelImg.py
  
  It's neccesary to obtain the label files (.txt) from dataset images.
  
  Then, execute jupyter notebook from anaconda y open Yolov3Files.ipyb to generate the train.txt and valid.txt files.
  
  $ jupyter notebook
  
  Finally you need to obtain the following files: 
  
  	In data/custom/images the dataset images
  
  	In data/custom/labels the dataset labels (.txt) 
  
  	In data/ the train.txt and valid.txt files
  
  	In data/ the classes.names file with the class names of dataset
  
  	In the folder config/ the file .cfg that corresponds to the class number of our dataset (FinBenthic)
  
  	In the folder config/ to modify the custom.data file with the class number of dataset (FinBenthic)
  
  To the file .cfg we need to open the bash (.sh) file and modify NUM_CLASSES=5 parameter. This file needs to execute using Git for Windows
  (Installer available at: https://git-scm.com/download/win)

  **Classes:**

  	l19: Legs of class 19 (Elmis_aena_adult)
  	h19: Head of class 19 (Elmis_aena_adult)
  	c19: Coxa of class 19 (Elmis_aena_adult)
  	e19: Elytra of class 19 (Elmis_aena_adult)
  	a19: Antenna of class 19 (Elmis_aena_adult)
  
  
### Train using Google Colab
  	$!pip install torch==1.1 torchvision==0.3
  
  	$!pip install opencv-python numpy matplotlib tensorboard terminaltables pillow tqdm
  
  	$!git clone https://github.com/liliandayanacruz/YoloMacroinvertebrates.git
  
  	$cd Yolov3Custom
  
  	$import urllib.request

  	$urllib.request.urlretrieve('https://pjreddie.com/media/files/darknet53.conv.74','/content/YoloMacroinvertebrates/weights/darknet53.conv.74')
	
	$from google.colab import drive
        
	$drive.mount('/content/drive')
	
	$!cp -r "/content/drive/My Drive/2024-1/YoloMacroinvertebrates/custom" "/content/YoloMacroinvertebrates/data"
        
	$!cp -r "/content/drive/My Drive/2024-1/YoloMacroinvertebrates/config" "/content/YoloMacroinvertebrates"

  Modify the file /usr/local/lib/libpackages/torchvision/transforms/functional.py
  
  Change this code line:
  
	from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
	
  For this
  
    from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
    
  Train
	
  	$!python train.py --model_def config/yolov3-custom5C.cfg --data_config config/custom.data --epochs 200 --batch_size 4 --pretrained_weights weights/darknet53.conv.74
  
### Local test in images sample

   Download the last file .pth and copy in the folder checkpoints local
   
   	python detectC.py --image_folder data/samplesC/ --model_def config/yolov3-custom5C.cfg --weights_path checkpoints/yolov3_ckpt_199.pth --class_path data/custom/classes.names
   
   	It's possible to test in a video with the follow line code:
    
    python detect_cam.py --model_def config/yolov3-custom5C.cfg --weights_path checkpoints/yolov3_ckpt_252.pth --class_path data/custom/classes.names --conf_thres 0.6
   

### Based on work by Erik Linder and David Revelo Luna (Youtube)
   https://github.com/eriklindernoren
   
   https://www.youtube.com/watch?v=JGS-FopcNyA

### Original dataset
   https://www.tooploox.com/blog/card-detection-using-yolo-on-android
