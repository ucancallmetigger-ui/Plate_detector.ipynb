# Plate_detector.ipynb
Iranian Vehicle License Plate Detection & Recognition

This project implements an end-to-end vehicle license plate detection and recognition system specifically designed for Iranian license plates. The pipeline combines a YOLOv8-based object detection model with a deep learning OCR model, supported by classical image preprocessing techniques to enhance robustness and accuracy.

License plate recognition is addressed as a two-stage computer vision problem:

Detection – Locating the license plate region in the image

Recognition – Reading and decoding the characters on the detected plate

This project performs both tasks sequentially in a fully automated pipeline.

Models Used

>>License Plate Detection

>Model: YOLOv8 (Medium)

>Checkpoint: iran_plate_detection.pt

>Source: Hugging Face

>Purpose: Accurate localization of Iranian vehicle license plates in complex scenes

YOLOv8 is chosen due to its:

-High detection accuracy

-Fast inference speed

-Robust performance under varying lighting and backgrounds

>>License Plate OCR

>Model: crnn-fa-license-plate-recognition-v2

>Framework: Hezar

>Architecture: CRNN (CNN + RNN + CTC)

>Language Support: Persian (Farsi)

The CRNN model is well-suited for license plate recognition because it:

-Extracts visual features using CNNs

-Models character sequences using recurrent layers

-Handles variable-length text without explicit character segmentation

>>Image Preprocessing Pipeline

Before detection and recognition, each image is passed through a preprocessing pipeline to improve visual clarity and OCR performance.

>>Grayscale Conversion

-cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

>Removes unnecessary color information

>Reduces computational complexity

>Emphasizes structural and textual features

>>Histogram Equalization

-cv2.equalizeHist(gray)

>Improves contrast in low-light or high-glare images

>Enhances character visibility

>Makes plate regions more distinguishable

>>Noise Reduction

-cv2.fastNlMeansDenoising(equalized, h=20)

>Removes random noise while preserving edges

>Improves stability of both detection and OCR

>Particularly useful for real-world images captured by cameras

>>Image Enhancement

-cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

>Converts the processed image back to 3 channels

>Ensures compatibility with YOLOv8 input requirements

-Impact:
This preprocessing pipeline significantly improves detection confidence and OCR accuracy, especially under challenging lighting conditions.

>>Detection & Recognition Workflow

The complete pipeline operates as follows:

>Install required dependencies (ultralytics, hezar, opencv)

>Load the pretrained YOLOv8 detection model

>Load the CRNN-based OCR model

>Read the input image

>Apply preprocessing steps

>Detect the license plate bounding box

>Crop the detected plate region

>Perform OCR on the cropped plate

>Draw bounding box and recognized text on the image

>Save and display the final result

Key Implementation Details
>>Model Integration

-YOLOv8 handles spatial localization

-CRNN handles sequential text recognition

-Models are decoupled for modularity and easy replacement

>>Bounding Box Selection

-box = results.boxes.xyxy[0]

>The most confident detected plate is selected

>Simplifies inference for single-plate images

>>Output Visualization

-Bounding boxes and text are drawn using OpenCV

-Results are saved as an output image

-Visualized directly inside Google Colab

>>Dependencies

>Python

>Ultralytics (YOLOv8)

>Hezar

>OpenCV

>Google Colab utilities

 

