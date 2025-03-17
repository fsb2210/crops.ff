# TODO
---

This document outlines the steps to implement a two-stage approach for detecting plants and classifying them as "healthy" or
"non-healthy" using external datasets

---

## 1. Dataset preparation

### 1.1. Download the PlantDoc dataset
- [x] Download the PlantDoc dataset from the official source
- [x] Explore the dataset structure (images and annotations)

### 1.2. Prepare object detection dataset
- [x] Convert annotations to YOLO format (or another format compatible with your object detection model)
- [x] Organize the dataset into the following structure:

```
plantdoc_detection/
├── images/
│ ├── train/
│ ├── val/
│ └── test/
└── labels/
  ├── train/
  ├── val/
  └── test/
```

### 1.3. Prepare classification dataset
- [ ] Crop bounding box regions from the images
- [ ] Label each cropped image as "healthy" or "non-healthy"
  - Example script: `crop_and_save.py`
- [ ] Organize the dataset into the following structure:

```
plantdoc_classification/
├── healthy/
└── non-healthy/
```

---

## 2. Train the object detection model

### 2.1. Choose an object detection model
- [ ] Select a model (e.g., YOLO, Faster R-CNN, or SSD)

### 2.2. Set pp the training environment
- [ ] Install required libraries (e.g., `ultralytics` for YOLO, `pytorch` for Faster R-CNN)
- [ ] Prepare the configuration file (e.g., `plantdoc.yaml` for YOLO)

### 2.3. Train the model
- [ ] Train the object detection model on the PlantDoc dataset
- [ ] Evaluate the model using metrics like mAP (mean Average Precision)

### 2.4. Save the trained model
- [ ] Export the trained model for inference (e.g., `best.pt` for YOLO)

---

## 3. Train the classification model

### 3.1. Choose a classification model
- [ ] Select a model (e.g., EfficientNet, ResNet, or MobileNet)

### 3.2. Set up the training environment
- [ ] Install required libraries (e.g., `pytorch`)
- [ ] Prepare the dataset for training (e.g., split into train/val/test sets)

### 3.3. Train the model
- [ ] Train the classification model on the cropped images
- [ ] Evaluate the model using metrics like accuracy, precision, recall, and F1-score

### 3.4. Save the trained model
- [ ] Export the trained model for inference (e.g., `classification_model.h5`)

---

## 4. Implement the inference pipeline

### 4.1. Load the trained models
- [ ] Load the object detection model (e.g., `best.pt` for YOLO)
- [ ] Load the classification model (e.g., `classification_model.h5`)

### 4.2. Write the inference script
- [ ] Write a script to:
  1. Detect plants in an image using the object detection model
  2. Crop the detected regions
  3. Classify each crop as "healthy" or "non-healthy" using the classification model

### 4.3. Test the pipeline
- [ ] Test the pipeline on sample images from the PlantDoc dataset
- [ ] Visualize the results (e.g., draw bounding boxes and labels on the image)

---

## 5. Optimize and deploy

### 5.1. Optimize the models
- [ ] Fine-tune hyperparameters for better performance
- [ ] Use techniques like data augmentation and regularization to reduce overfitting

### 5.2. Deploy the pipeline
- [ ] Export the models to a deployable format (e.g., ONNX)
- [ ] Deploy the pipeline on a platform (e.g., Flask/Django for web)

---

## 6. Documentation and Reporting

### 6.1. Document the process
- [ ] Include instructions for dataset preparation, training, and inference

### 6.2. Report results
- [ ] Create a report summarizing the results (e.g., accuracy, mAP, inference speed)
- [ ] Include visualizations of sample predictions

---

## 7. Future improvements
- [ ] Experiment with end-to-end models (e.g., Mask R-CNN)
- [ ] Add support for more plant species and diseases
- [ ] Optimize for real-time inference on edge devices

---

## Tools and libraries
- **Object Detection**: YOLO, Faster R-CNN, SSD
- **Classification**: EfficientNet, ResNet, MobileNet
- **Frameworks**: PyTorch
- **Data Processing**: OpenCV, NumPy, Pandas

---

## Contributors
- fsb2210

---

## **License**
- Some open source one...
