# Multimodal Sentiment Classifier
This is Xu Hanyi's repository for the Fifth Experiment in class _Contemporary Artifical Intelligence_, 2023.

## Setup
This implemetation is based on Python3. To run the code, you need the following dependencies:
- nltk==3.7

- Pillow==10.0.0

- torch==2.0.1

- torchtext==0.15.2

- torchvision==0.10.0

You can simply run 

```python
pip install -r requirements.txt
```

## Repository structure
We select some important files for detailed description.

```python

|-- datafolder # data to be trained and tested, not yet uploeaded
    |-- data/ # folder for texts and images, each sample has a unique text and a unique image, 4000 samples in sum
    |-- train.txt # labels
    |-- test_without_label.txt # samples to be tested
|-- main.py # the main code
|-- main_txt_only.py # models with only texts as input 
|-- main_img_only.py # models with only images as input 
```

## Run pipeline for big-scale datasets
You can run any models implemented in 'models.py'. For examples, you can run our model on 'genius' dataset by the script:
```python
python main.py
```
And you can run other models, such as 
```python
python main_img_only.py
```
