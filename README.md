# Digit Recognizer using PyTorch

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset using PyTorch.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Submission](#submission)
- [Results](#results)
- [Usage](#usage)
- [References](#references)

## Introduction

This project demonstrates the use of a CNN to classify handwritten digits from the MNIST dataset. The dataset consists of images of digits ranging from 0 to 9, and the model is trained to classify these images correctly.

## Dataset

The dataset used in this project is the MNIST dataset, which is available on Kaggle. The training data consists of 42,000 images, and the test data consists of 28,000 images.

## Installation

To run this project, you need to have Python and the following libraries installed:

- numpy
- pandas
- matplotlib
- scikit-learn
- torch
- torchvision
- torchsummary

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib scikit-learn torch torchvision torchsummary
```

## Model Architecture

The model architecture consists of the following layers:

- Two convolutional layers with batch normalization and ReLU activation
- Max-pooling layers
- Fully connected layers

Here is a summary of the model:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             832
       BatchNorm2d-2           [-1, 32, 26, 26]              64
         MaxPool2d-3           [-1, 32, 13, 13]               0
            Conv2d-4           [-1, 64, 11, 11]          51,264
       BatchNorm2d-5           [-1, 64, 11, 11]             128
         MaxPool2d-6             [-1, 64, 5, 5]               0
            Linear-7                  [-1, 256]         409,856
            Linear-8                   [-1, 10]           2,570
================================================================
Total params: 464,714
Trainable params: 464,714
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.50
Params size (MB): 1.77
Estimated Total Size (MB): 2.28
----------------------------------------------------------------
```

## Training

The model is trained using the Adam optimizer with a learning rate of 0.001 and a weight decay of 1e-4. The training process includes:

- Forward pass
- Loss computation using cross-entropy loss
- Backward pass
- Parameter update

```python
for epoch in range(cfg.epochs):
    running_loss = 0.0
    for img, label in train_loader:
        img = img.to(cfg.device)
        label = label.to(cfg.device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch + 1} loss: {running_loss / len(train_loader)}")
```

## Evaluation

The model's accuracy on the training set is evaluated after training:

```python
correct = 0
total = 0
with torch.no_grad():
    for img, label in train_loader:
        img = img.to(cfg.device)
        label = label.to(cfg.device)
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
  
    print(f"Accuracy: {100 * correct / total}%")
```

## Submission

The model's predictions on the test set are generated and saved in a CSV file for submission:

```python
prediction = []
with torch.no_grad():
    for img in test:
        img = img.to(cfg.device)
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        prediction.extend(predicted.cpu().numpy())
submission = pd.DataFrame({'ImageId': list(range(1, len(prediction) + 1)), 'Label': prediction})
submission.to_csv('submission.csv', index=False)
```

## Results

The model achieves an accuracy of approximately 99.99% on the training set. Detailed results and analysis can be found in the notebook.

## Usage

To use this code:

1. Clone the repository
2. Install the required libraries
3. Download the dataset from Kaggle
4. Run the script

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Kaggle MNIST Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)

Feel free to explore the code and modify it as needed. For any questions or suggestions, please open an issue or contact me directly.

## Author

Shubham Vats  
[shubhamvats02@gmail.com](mailto:shubhamvats02@gmail.com)  
[GitHub](https://github.com/shubvats)
