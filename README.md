# cnn_circle_detection
This is a PyTorch implementation of a Deep Convolutional Neural Network model for detecting circle objects in images.

# Network Architecture

The output of the network is 4 real numbers which represents the detected center coordinates x,y,width and height.

# Getting Started

**Installation**
- Clone this repo:
```shell
git clone https://github.com/wyattPol/circle_detection_cnn
```

- Requirements
Install the dependencies by running the following command:
```shell
pip install -r requirements.txt
```

# Model training
- Data set
[here is dataset source]

```shell
https://universe.roboflow.com/trees-m4e49/-39p4p/dataset/1
```

- Training:

```shell
python train3.py
```

- Resume training from a saved model:

You are able to resume training your model from a saved checkpoint by running the following:

```shell
python train.py --resume {directory path to your saved model}
```
# Model Testing

THere is already the test.py you can use(be careful with path):

```shell
python test.py
``` 
In this repo you can see 2 network, folder model_net is the trained model from network.py.
folder model_net_1 contains the trained model from network2.py.

# Results on examples

Display like below 

![circle_detect](images/Image -2-_png_jpg.rf.3e95045430f4150989887ab0ab8b21ce.jpg with Bounding Box_screenshot_16.05.2024.png)

![circle_detect](images/Image -32-_png_jpg.rf.9d9559d190bce5e91f27ea44b2feecf1.jpg with Bounding Box_screenshot_16.05.2024.png)
![circle_detect](images/Image Ekran-goruntusu-2024-04-17-191620_png_jpg.rf.5d9a6ee3867aa5849210ddbee033afd2.jpg with Bounding Box_screenshot_16.05.2024.png)
![circle_detect](images/Image frame419_jpg.rf.3b3d4a3164b9c83910d568726aa21566.jpg with Bounding Box_screenshot_16.05.2024.png)
![circle_detect](images/Image images_jpg.rf.c6dcdbcc89e254ad3493a6be1b675561.jpg with Bounding Box_screenshot_16.05.2024.png)
