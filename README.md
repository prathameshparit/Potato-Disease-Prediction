# Potato Plant Disease Prediction



Potato is one of the most significant food crops. The diseases cause substantial yield loss in potato. Early detection of these diseases can allow to take preventive measures and mitigate economic and production losses. Over the last decades, the most practiced approach for the detection and identification of plant diseases is naked eye observation by experts. But in many cases, this approach proves unfeasible due to the excessive processing time and unavailability of experts at farms located in remote areas. Hence the problem statement for research work is defined as “AI BASED CROP HEALTH MANAGEMENT SYSTEM TO INCREASE THE PRODUCE”

## Website:
![Gif](https://github.com/prathameshparit/Potato-Disease-Prediction/blob/bff4d32202091819bedf1844b75316b3e7b504b2/Potato-Disease.gif?raw=true)


### The project includes these points for potato disease detection:
- Comparative analysis of various disease detection methods for automated potato crop disease detection.
- Prepared a dataset for early detection of various potato crop diseases using python with ML.
- A novel model for early disease diagnosis in potato crops for appropriate auto-notification of remedy.

The predicted input image is then classified into following classes:

- **Early Blight** - Early blight of potato is caused by the fungal pathogen Alternaria solani. The disease affects leaves, stems and tubers and can reduce yield, tuber size, storability of tubers, quality of fresh-market and processing tubers and marketability of the crop.


![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/95a67f825a4e8932a6caa28f9147a8a4dee0af66/readme%20images/Upload/Early_Blight_87.jpg?raw=true)


- **Healthy** - Healthy is the healthy state of a leaf where it is not affected with any kind of disease which leads to a proper and healthy growth of potato in future stages of it's evolution.


![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/95a67f825a4e8932a6caa28f9147a8a4dee0af66/readme%20images/Upload/Healthy_5.jpg?raw=true)

- **Late Blight** - Late blight caused by the fungus Phytophthora infestans is the most important disease of potato that can result into crop failures in a short period if appropriate control measures are not adopted. Losses in potato yield can go as high as 80% in epidemic years.


![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/95a67f825a4e8932a6caa28f9147a8a4dee0af66/readme%20images/Upload/Late_Blight_98.jpg?raw=true)

## Exploratory Data Analysis

- 1303 Early Blight Total number of images model is trained on Early Blight

- 816 Healthy Total number of images model is trained on Healthy

- 1132 Late Blight Total number of images model is trained on Late Blight

## Data Visualization
![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/95a67f825a4e8932a6caa28f9147a8a4dee0af66/readme%20images/Upload/visualizing.png?raw=true)

## Features

- Drag and drop images 
- Drop images to predict Potato Plant Disease
- It predicts the input image for 3 different classes(Early Blight, Healthy, Late)
- Predicts the accuracy with 7 different classifiers for the input image
- Each Classifier provides the metrics for particular prediction with graph



## Tech

The website uses a number of open source projects to work properly:

- [Tensorflow] - Deep learning application framework
- [Scikit-Learn] - Bank for classification, predictive analytics, and very many other machine learning tasks.
- [Flask] - Framework for creating web applications in Python easier.
- [Matplotlib] - A low level graph plotting library in python that serves as a visualization utility.
- [Numpy] - Used for working with arrays
- [Pandas] - Used for data analysis and associated manipulation of tabular data in Dataframes

## Screenshots and Steps

**1. Landing Page:**

- This is the landing page for the web application 

- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/95a67f825a4e8932a6caa28f9147a8a4dee0af66/readme%20images/Upload/landing.png?raw=true)

**2. Upload button:**
 
- Later on the web application it provides 3 different buttons along with a upload button where you upload your input image of potato plant disease and later it provides you with 3 buttons of Preprocessing, Feature Extraction and Prediction of the uploaded image

- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/95a67f825a4e8932a6caa28f9147a8a4dee0af66/readme%20images/Upload/upload.png?raw=true)

**3. Preprocessing:**


- After uploading the image the image needs to be preprocessed where it is preprocessing using two techniques which is Resizing of the uploaded input image from it's original size to the size which is required for the image to predict on.

- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/95a67f825a4e8932a6caa28f9147a8a4dee0af66/readme%20images/Upload/prepro.png?raw=true)

- After resizing of the image Data Augmentation is applied on the input image  
- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/95a67f825a4e8932a6caa28f9147a8a4dee0af66/readme%20images/Upload/dataaug.png?raw=true)

**5. Feature Extraction :**

- After Preprocessing comes the part of Feature Extraction where we extract important features of the input image by converting the uploaded image from RGB to HSV and pointing out the important parts required for the model to predict the following image

- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/95a67f825a4e8932a6caa28f9147a8a4dee0af66/readme%20images/Upload/hsv.png?raw=true)

**6. Classifiers :**

- After the Feature Extraction comes the part of Prediction where the project is trained on 7 different classifiers which are SVM, KNN, ANN, DT, CNN, Hybrid(SVM+ANN), Hybrid2(CNN+KNN) and it displays it's prediction on those 7 classifiers along with the confidence at which it has predicted the following image 
- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/95a67f825a4e8932a6caa28f9147a8a4dee0af66/readme%20images/Upload/classi.png?raw=true)

- If you click on any of the classifiers it further shows you the classification metrics on that particular classifier along with the visualization of that model
-![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/95a67f825a4e8932a6caa28f9147a8a4dee0af66/readme%20images/Upload/metrics.png?raw=true)

**7. Comparitive Analysis :**
- At last the application provides you with a comparitive analysis of all the classifiers where you can compare the accuracy of each classfier side by side in the format of table as well as graph
- ![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/95a67f825a4e8932a6caa28f9147a8a4dee0af66/readme%20images/Upload/comparitive.png?raw=true)

## Results

The following project has shown some promising results for classifying the potato plant diseases into 3 classes which is Early Blight, Healthy, Late Blight and here's the classification for the following
![App Screenshot](https://github.com/prathameshparit/Dummy-Storage/blob/95a67f825a4e8932a6caa28f9147a8a4dee0af66/readme%20images/Upload/conf.png?raw=true)




## Installation

Website requires these steps to install the application on your device


On terminal:

Download virtual env library
```sh
pip3 install -U pip virtualenv
```

Create a virtual environment on your device
```sh
virtualenv  -p python3 ./venv
```

Download all the dependencies provided on requirements.txt
```sh
pip install -r .\requirements.txt
```

Activated the virtual environment
```sh
.\pp\Scripts\activate
```

Run app.py after completing all the steps.





[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   
[Tensorflow]: <https://www.tensorflow.org/>
[Scikit-Learn]: <https://scikit-learn.org/stable/>
[Flask]: <https://flask.palletsprojects.com/en/2.1.x/>
[Matplotlib]: <https://matplotlib.org/>
[Numpy]: <https://numpy.org/>
[Pandas]: <https://pandas.pydata.org/>


