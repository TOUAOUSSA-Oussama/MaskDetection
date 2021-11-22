# MaskDetection
This application allows the detection of faces in an image and then predicts the existence of a mask for each face. It allows you to make the prediction in real time by activating your camera and then making the prediction on each frame.

The utility of this application is to fight against COVID-19 by controlling the respect of the wearing of the mask. It can be taken on board surveillance cameras and used in any area that requires a mask to be worn.

The prediction is made using the MobileNet tensorflow model. Detection of faces in each frame / frame is done using the OpenCV library.

The results of training the MobileNet model are shown in the following figure:

------------------------------

To run the project:
- Install the dependencies: pip install -r requirements.txt
- If you want to re-train the model: python train_mask_detector.py
- If you want to run the application: python detect_mask_video.py


Example :
-----------------------------------
