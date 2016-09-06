# kaggle-distracted-driver
This is the classifier that I built for Kaggle Distracted Driver challange. I used R-CNN to detect different objects in the image and then derived features from them. Features included both boolean features(object present/not) as well as features on objects' position. The final classifier uses these features to predict one of the 10 classes. 
The code here is not standalone. It depends on detected objects from R-CNN. 
Very small portion of the training set is manually annotated with bounding boxes for target objects. The R-CNN is tuned for this target objects.
