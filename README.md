# Automated Essay Scoring
> A Deep Learning model that predicts the score of a given input essay. 

The dataset is from Kaggle ASAP competition which was provided by The Hewlett Foundation.

The **mysite** folder contains the Django app if you want an interactive demo.

### Performance
The accuracy is calculated by **Quadratic Weighted Kappa(QWK)**, which measures the agreement between two raters. The state of the art implementation in this competition achieved a **QWK score of 0.82** six years ago. My model achieves a **QWK score of 0.961**. The model architecture consists of 2 Long Short Term Memory(LSTM) layers with a Dense output layer. The final layer uses the Relu activation function. The QWK is calculated by training model on the dataset using 5-Fold Cross Validation and taking the average for all five folds.

For more info on LSTM Networks see this link.

![](https://github.com/mankadronit/Automated-Essay--Scoring/blob/master/ScreenShots/SC1.png = 250x250)

![](https://github.com/mankadronit/Automated-Essay--Scoring/blob/master/ScreenShots/SC2.png = 250x250)

![](https://github.com/mankadronit/Automated-Essay--Scoring/blob/master/ScreenShots/SC3.png = 250x250)

## Requirements
- Tensorflow 1.9
- Keras 2.2.2
- Django 2.1
- Gensim 3.5

```
pip install tensorflow
pip install keras
pip install django
pip install gensim

```

## Installation 
- Clone the repo
- Just run the **Jupyter Notebook** to train the model.
- To run the Django App cd ./ into the **mysite** folder and run ```python manage.py runserver```

## Reference
(A Neural Approach to Automated Essay Scoring)[http://aclweb.org/anthology/D/D16/D16-1193.pdf]
(Automatic Text Scoring Using Neural Networks)[https://arxiv.org/pdf/1606.04289.pdf]
