# Deepfake-Detection-Project
Deep learning-based deepfake detection using MobileNetV2 for real vs fake image classification.
# Deepfake Detection using Deep Learning

## Project Description
This project uses MobileNetV2 to detect deepfake images by classifying them as real or fake. Video frames are extracted using OpenCV and processed through preprocessing steps like resizing and normalization before training the model.

## Methodology
- Frame extraction from videos using OpenCV  
- Image preprocessing (resizing to 224×224, normalization)  
- Transfer learning using MobileNetV2  
- Binary classification (real vs fake images)  

## Results
- Achieved accuracy of 88–89%  
- Model performs well on the given dataset  

## How to Run
```bash
pip install -r requirements.txt
python train_cnn.py
## Output Samples

### Predictions
![Predictions](Predictions .png)

### Confusion Matrix
![Confusion Matrix](Confusion_Matrix.png)
