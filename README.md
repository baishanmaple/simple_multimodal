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
## Experiment Process
### 1. Dataset Preparation

The dataset used in the experiment is located in the `data` folder at the specified path. The dataset consists of two types of files: text files (ending with ".txt") and image files (ending with ".jpg"). The text files contain the content of the text to be classified, while the image files contain images associated with the text content.

### 2. Custom Dataset

In the experiment, a custom dataset class called `CustomDataset` is created. This class reads text files and image files, combines the text content, sentiment labels, and image paths into tuples, and uses them as samples for the dataset. Additionally, the text is tokenized using the `NLTK` library, and a vocabulary is built.

### 3. Model Design

The experiment utilizes a deep learning model called `SentimentClassifier`. This model performs embedding operations on the text and image data separately, and then extracts features from the images using a convolutional neural network. The text and image features are then concatenated and further processed through fully connected layers, LSTM layers, and hidden layers to ultimately output the probabilities of sentiment categories. 

Here is the code for the model:

```python
class SentimentClassifier(nn.Module):
    def __init__(self, text_input_size, image_input_size, hidden_size, num_classes):
        super(SentimentClassifier, self).__init__()

        # Text modality layers
        self.text_embedding = nn.Embedding(text_input_size, 128)
        self.text_lstm1 = nn.LSTM(128, hidden_size, batch_first=True)
        self.text_lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Image modality layers
        self.image_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.image_conv2 = nn.Conv2d(64, 128, stride=2, kernel_size=3)
        self.image_bn1 = nn.BatchNorm2d(128)
        
        self.image_conv3 = nn.Conv2d(128, 32, kernel_size=3, stride=2, padding=1)
        self.image_bn2 = nn.BatchNorm2d(32)
        self.image_fc = nn.Linear(32*28*28, hidden_size)

        # Fusion layers for combining the two modalities
        self.fusion_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p=0.5)

        # Final layer
        self.output_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, text_input, image_input):
        # Forward pass for the text modality
        text_output = self.text_embedding(text_input)
        text_output, _ = self.text_lstm1(text_output)
        text_output = self.dropout(text_output)
        text_output, _ = self.text_lstm2(text_output)

        # Forward pass for the image modality
        image_output = self.image_conv1(image_input)
        image_output = self.dropout(image_output)
        image_output = torch.relu(image_output)
        
        image_output = self.image_conv2(image_output)
        image_output = self.dropout(image_output)
        image_output = torch.relu(image_output)
        image_output = self.image_bn1(image_output)
        
        image_output = self.image_conv3(image_output)
        image_output = self.dropout(image_output)
        image_output = self.image_bn2(image_output)
        image_output = image_output.view(image_output.size(0), -1)
        image_output = torch.relu(self.image_fc(image_output))

        # Forward pass for the fusion of the two modalities
        fused_output = torch.cat((text_output[:, -1, :], image_output), dim=1)
        fused_output = self.dropout(fused_output)
        fused_output = self.fusion_fc(fused_output)
        fused_output = self.dropout(fused_output)
        fused_output = torch.relu(fused_output)
        
        # Final layer
        output = self.output_fc(fused_output)
        return output
```

### 4. Model Structure Analysis:

- Text Modality Layers:
  - `self.text_embedding` is an embedding layer that maps text indices to dense vector representations.
  - `self.text_lstm1` is an LSTM layer that processes the text sequence and captures contextual information.
  - `self.text_lstm2` is the second LSTM layer that further processes the text features.

- Image Modality Layers:
  - `self.image_conv1` is a 2D convolutional layer that extracts local features from the images.
  - `self.image_conv2` is another 2D convolutional layer that further extracts image features.
  - `self.image_bn1` is a batch normalization layer that accelerates the training process and stabilizes the convergence of the model.
  - `self.image_conv2` is the third 2D convolutional layer that further compresses the image features.
  - `self.image_bn2` is another batch normalization layer that normalizes the feature maps.

- Fusion Layers for Combining the Two Modalities:
  - `self.fusion_fc` is a fully connected layer that combines the text and image features.
  - `self.dropout` is a dropout layer that randomly drops out a portion of the neuron outputs during training to reduce overfitting.

- Final Layer:
  - `self.output_fc` is a fully connected layer that maps the fused features to a probability distribution over sentiment categories.

In the forward pass, the text input and image input are passed through their respective modality layers for processing. Then, features are extracted from the last layer of each modality, and they are concatenated along the feature dimension. The concatenated features are passed through the fusion layer, and then through the final layer to obtain the final output, which represents the probability distribution over sentiment categories.
