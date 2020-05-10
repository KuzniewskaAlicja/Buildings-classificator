# **Buildings_classifier**

In the repository there is a final project (the first of two) made during advanced image processing classes at the Poznan University of Technology. The aim of the project was the classification of the five most most popular buildings in Poznan based on Bags of visual words.

Class names of considered buildings:
|Baltyk|Katedra|Okraglak|Uam|Teatr Wielki|
|:----:|:-----:|:------:|:-:|:----------:|
|<img src=".\dataset\baltyk\5e4160d6e7b4ef793bcf08b4__JKS7237.jpg" width = 400>|<img src=".\dataset\katedra\1Katedra.png" width = 400>|<img src=".\dataset\okraglak\305122.jpg" width = 350>|<img src=".\dataset\uam\5b0e61da30ca9_p.jpg" width = 500>|<img src=".\dataset\teatr_wielki\272098.jpg" width = 400>|

## **Requirements**
- python 3.7.5
- numpy 1.18.3
- opencv-python 4.2.0
- scikit-learn 0.22.2.post1

## **Result**
The best result on this test dataset was obtained for SVM classifier. 
Image width have been set to 700px and number of words in dictionary to 440.

Final accuracy and confusion matrix:

|              |  Accuracy [%]  |
|:------------:|:--------------:|
| Test data    |    89.41       | 
| Train data   |     100        | 

![Confusion Matrix](./chart/confusion_matrix.png)
