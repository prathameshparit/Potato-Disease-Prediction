# Potato Plant Disease Prediction



Potato is one of the most significant food crops. The diseases cause substantial yield loss in potato. Early detection of these diseases can allow to take preventive measures and mitigate economic and production losses. Over the last decades, the most practiced approach for the detection and identification of plant diseases is naked eye observation by experts. But in many cases, this approach proves unfeasible due to the excessive processing time and unavailability of experts at farms located in remote areas. Hence the problem statement for research work is defined as “AI BASED CROP HEALTH MANAGEMENT SYSTEM TO INCREASE THE PRODUCE”

## Demo:

https://user-images.githubusercontent.com/63944541/209446705-8dc40b11-08bd-4b68-b32d-f9fb69b2ae66.mov


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


