<p align="center">
 <h1 align="center">Fake New Detection</h1>
</p>

## Introduction
This project aims to build a fake news detection model using machine learning techniques. The model analyzes the textual content and subject of news articles to classify them as either "Fake" or "True".

## Dataset
The project utilizes two datasets: one containing fake news articles and the other containing true news articles.

You can find this dataset at <a href="https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets">this link</a>

## Training
You can find at **train.py** and the trained model is at **trained model.pkl**

## Evaluation
### Classification Report
||precision|recall|f1-score|support|  
|-----------|:-----------:|:-----------:|:-----------:|:-----------:|
|Fake|1.00|1.00|1.00|9420|
|True|1.00|1.00|1.00|8540|
|accuracy|||1.00|17960|
|macro avg|1.00|1.00|1.00|17960|
|weighted|1.00|1.00|1.00|17960|
### Confunsion Matrix
<img src="confusion_matrix.jpg" width="600" height="400">

## Requirements
* python
* pandas
* sklearn
* matplotlib
