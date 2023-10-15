## **Deep Learning Image Classification Model**

### **1 - Import Libraries**

Torch API


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchsummary import summary
from tqdm.notebook import tqdm
```

Data API


```python
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import re
import requests
```

Install opendatasets package


```python
pip install opendatasets
```

    Requirement already satisfied: opendatasets in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (0.1.22)
    Requirement already satisfied: kaggle in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (from opendatasets) (1.5.16)
    Requirement already satisfied: click in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (from opendatasets) (7.1.2)
    Requirement already satisfied: tqdm in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (from opendatasets) (4.50.2)
    Requirement already satisfied: requests in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (from kaggle->opendatasets) (2.31.0)
    Requirement already satisfied: bleach in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (from kaggle->opendatasets) (3.2.1)
    Requirement already satisfied: six>=1.10 in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (from kaggle->opendatasets) (1.15.0)
    Requirement already satisfied: certifi in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (from kaggle->opendatasets) (2022.6.15)
    Requirement already satisfied: python-slugify in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (from kaggle->opendatasets) (8.0.1)
    Requirement already satisfied: urllib3 in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (from kaggle->opendatasets) (1.25.11)
    Requirement already satisfied: python-dateutil in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (from kaggle->opendatasets) (2.8.2)
    Requirement already satisfied: idna<4,>=2.5 in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (from requests->kaggle->opendatasets) (2.10)
    Requirement already satisfied: charset-normalizer<4,>=2 in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (from requests->kaggle->opendatasets) (3.1.0)
    Requirement already satisfied: webencodings in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (from bleach->kaggle->opendatasets) (0.5.1)
    Requirement already satisfied: packaging in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (from bleach->kaggle->opendatasets) (20.4)
    Requirement already satisfied: text-unidecode>=1.3 in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (from python-slugify->kaggle->opendatasets) (1.3)
    Requirement already satisfied: pyparsing>=2.0.2 in /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages (from packaging->bleach->kaggle->opendatasets) (2.4.7)
    Note: you may need to restart the kernel to use updated packages.



```python
import opendatasets as od
```

### **2 - Download the Data**


```python
url = 'https://www.kaggle.com/trolukovich/apparel-images-dataset'
```


```python
od.download(url)
```

    Skipping, found downloaded files in "./apparel-images-dataset" (use force=True to force download)


### **3 - Get the Path of Images and the Labels (from directory name)**


```python
def get_path_names(dir):
  images = []
  labels = []

  for path, subdir, files in os.walk(dir):
    for fname in files:
      image_path = os.path.join(path, fname)
      images.append(image_path)
      label = list(os.path.join(path).split('/')[2].split('_')[:])
      labels.append(label)

  return images, labels

images, labels = get_path_names('./apparel-images-dataset')
```

### **4 - Setup the Output Classes (to be used for encoding and decoding)**


```python
classes = ['black',
           'blue',
           'brown',
           'green',
           'white',
           'red',
           'dress',
           'pants',
           'shorts',
           'shoes',
           'shirt']
```

### **5 - Define the Encoder and Decoder** (convert target to label and vice versa)


```python
def encode(label, classes=classes):
  output = torch.zeros(11)
  for l in label:
    idx = classes.index(l)
    output[idx] = 1
  return output

def decode(output, threshold=0.5):
  label = []
  for i, n in enumerate(output):
    if (n >= threshold):
      label.append(classes[i])
  return ' '.join(label)
```

### **6 - Define Custom Dataset**


```python
class MyDataset(Dataset):
  def __init__(self, root_dir, transform=None):
    super().__init__()
    self.root_dir = root_dir
    self.transform = transform
    self.images, self.labels = get_path_names(root_dir)

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    image = Image.open(self.images[index])
    if self.transform:
      image = self.transform(image)
    label = self.labels[index]
    # print(label)
    return image, encode(label)

```

### **7 - Image Transformation**


```python
imagenet_stats = ([0.485, 0.45, 0.406], [0.229, 0.224, 0.225])
```


```python
transform = T.Compose([T.Resize(128),
                       T.RandomCrop(128),
                       T.RandomHorizontalFlip(),
                       T.RandomRotation(2),
                       T.ToTensor(),
                       T.Normalize(*imagenet_stats)]);
```

### **8 - Load Dataset**


```python
dataset = MyDataset('./apparel-images-dataset', transform)
len(dataset)
```




    11385



### **9 - Denormalization Function to Show the Actual Image**


```python
def denorm(image_tensors):
  return image_tensors * imagenet_stats[1][0] + imagenet_stats[0][0]

def show_image(image, label):
  plt.imshow(denorm(image).permute(1,2,0))
  print("Label: ", decode(label))
  print(label)

show_image(*dataset[11380])
```

    Label:  blue shoes
    tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.])



    
![png](output_26_1.png)
    


### **10 - Train-Validation Split**


```python
val_size = int(0.15*len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)
```




    (9678, 1707)



### **11 - Define DataLoaders**


```python
batch_size = 32
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size*2)
```

### **12 - Define Utility Functions to load to device**


```python
def get_default_device():
  if torch.cuda.is_available():
    return torch.device('cuda')
  else:
    return torch.device('cpu')
```


```python
def to_device(data, device):
  if isinstance(data, (list, tuple)):
    return [to_device(x, device) for x in data]
  return data.to(device, non_blocking=True)
```

### **13 - Define Metrics**


```python
def F_score(output, label, threshold=0.5, beta=1):
  output = output > threshold
  label = label > threshold

  TP = (output & label).sum(1).float()
  TN = (~output & ~label).sum(1).float()
  FP = (output & ~label).sum(1).float()
  FN = (~output & label).sum(1).float()

  precision = torch.mean(TP/(TP + FP + 1e-12))
  recall = torch.mean(TP/(TP + FN + 1e-12))
  F2 = (1 + beta**2)* precision * recall / (beta**2 * precision + recall + 1e-12)

  return F2.mean(0)

```

### **14 - Define Traning and Validation STEP Routines**


```python
class MultiClassClassification(nn.Module):
  def train_step(self, batch):
    images, labels = batch
    output = self(images)
    loss = F.binary_cross_entropy(output, labels)
    return loss

  def validation_step(self, batch):
    images, labels = batch
    output = self(images)
    loss = F.binary_cross_entropy(output, labels)
    score = F_score(output, labels)
    return {'val_loss':loss.detach(), 'val_score':score.detach()}

  def validation_epoch_end(self, outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_scores = [x['val_score'] for x in outputs]
    epoch_score = torch.stack(batch_scores).mean()
    return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

  def epoch_end(self, epoch, result):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(epoch, result['train_loss'], result['val_loss'], result['val_score']))
```

### **15 - Define Training and Validation Process**


```python
def evaluate(model, val_dl):
  with torch.no_grad():
      model.eval()
      outputs = [model.validation_step(batch) for batch in val_dl]
      return model.validation_epoch_end(outputs)

def fit(epochs, max_lr, model, train_dl, val_dl, weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
  torch.cuda.empty_cache()
  history = []

  # Optimizer
  optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)

  for epoch in range(epochs):

    #Training
    model.train()
    train_losses = []
    # lrs = []

    for batch in tqdm(train_dl):
      loss = model.train_step(batch)
      train_losses.append(loss)
      loss.backward()

      # Gradient Clipping
      if grad_clip:
        nn.utils.clip_grad_value_(model.parameters(), grad_clip)

      optimizer.step()
      optimizer.zero_grad()

    # Validation
    result = evaluate(model, val_dl)
    result['train_loss'] = torch.stack(train_losses).mean().item()
    print(result)

    model.epoch_end(epoch, result)
    history.append(result)
            
  return history


```

### **16 - Model Creation**


```python
# Conv Block
def conv_block(input_channels, output_channels, pool=False):
  layers = [nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)]

  if pool: layers.append(nn.MaxPool2d(4))

  return nn.Sequential(*layers)
```


```python
# Project Model (ResNet15) based
class ProjectModel(MultiClassClassification):
  def __init__(self, in_channels, num_classes):
    super().__init__()

    # input 3 x 128 x 128

    self.conv1 = conv_block(in_channels, 64)
    self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64)) # 64 x 128 x 128

    self.conv2 = conv_block(64, 128, pool=True) # 128 x 32 x 32
    self.res2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

    self.conv3 = conv_block(128, 512, pool=True) # 512 x 8 x 8
    self.res3 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

    self.conv4 = conv_block(512, 1024, pool=True) # 1024 x 2 x 2
    self.res4 = nn.Sequential(conv_block(1024, 1024), conv_block(1024, 1024))

    self.classifier = nn.Sequential(nn.MaxPool2d(2), # 1024 x 1 x 1
                                    nn.Flatten(),
                                    nn.Dropout(0.2),
                                    nn.Linear(1024, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, num_classes))

  # Forward
  def forward(self, x):
    out = self.conv1(x)
    out = self.res1(out) + out
    out = self.conv2(out)
    out = self.res2(out) + out
    out = self.conv3(out)
    out = self.res3(out) + out
    out = self.conv4(out)
    out = self.res4(out) + out

    out = self.classifier(out)
    out = F.sigmoid(out)
    return out
```


```python
device = get_default_device()
model = to_device(ProjectModel(3, len(classes)), device)
model
```




    ProjectModel(
      (conv1): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (res1): Sequential(
        (0): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (1): Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
      )
      (conv2): Sequential(
        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
      )
      (res2): Sequential(
        (0): Sequential(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (1): Sequential(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
      )
      (conv3): Sequential(
        (0): Conv2d(128, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
      )
      (res3): Sequential(
        (0): Sequential(
          (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (1): Sequential(
          (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
      )
      (conv4): Sequential(
        (0): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
      )
      (res4): Sequential(
        (0): Sequential(
          (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (1): Sequential(
          (0): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
      )
      (classifier): Sequential(
        (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (1): Flatten(start_dim=1, end_dim=-1)
        (2): Dropout(p=0.2, inplace=False)
        (3): Linear(in_features=1024, out_features=512, bias=True)
        (4): ReLU()
        (5): Linear(in_features=512, out_features=11, bias=True)
      )
    )




```python
summary(model, (3, 128, 128))
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1         [-1, 64, 128, 128]           1,792
           BatchNorm2d-2         [-1, 64, 128, 128]             128
                  ReLU-3         [-1, 64, 128, 128]               0
                Conv2d-4         [-1, 64, 128, 128]          36,928
           BatchNorm2d-5         [-1, 64, 128, 128]             128
                  ReLU-6         [-1, 64, 128, 128]               0
                Conv2d-7         [-1, 64, 128, 128]          36,928
           BatchNorm2d-8         [-1, 64, 128, 128]             128
                  ReLU-9         [-1, 64, 128, 128]               0
               Conv2d-10        [-1, 128, 128, 128]          73,856
          BatchNorm2d-11        [-1, 128, 128, 128]             256
                 ReLU-12        [-1, 128, 128, 128]               0
            MaxPool2d-13          [-1, 128, 32, 32]               0
               Conv2d-14          [-1, 128, 32, 32]         147,584
          BatchNorm2d-15          [-1, 128, 32, 32]             256
                 ReLU-16          [-1, 128, 32, 32]               0
               Conv2d-17          [-1, 128, 32, 32]         147,584
          BatchNorm2d-18          [-1, 128, 32, 32]             256
                 ReLU-19          [-1, 128, 32, 32]               0
               Conv2d-20          [-1, 512, 32, 32]         590,336
          BatchNorm2d-21          [-1, 512, 32, 32]           1,024
                 ReLU-22          [-1, 512, 32, 32]               0
            MaxPool2d-23            [-1, 512, 8, 8]               0
               Conv2d-24            [-1, 512, 8, 8]       2,359,808
          BatchNorm2d-25            [-1, 512, 8, 8]           1,024
                 ReLU-26            [-1, 512, 8, 8]               0
               Conv2d-27            [-1, 512, 8, 8]       2,359,808
          BatchNorm2d-28            [-1, 512, 8, 8]           1,024
                 ReLU-29            [-1, 512, 8, 8]               0
               Conv2d-30           [-1, 1024, 8, 8]       4,719,616
          BatchNorm2d-31           [-1, 1024, 8, 8]           2,048
                 ReLU-32           [-1, 1024, 8, 8]               0
            MaxPool2d-33           [-1, 1024, 2, 2]               0
               Conv2d-34           [-1, 1024, 2, 2]       9,438,208
          BatchNorm2d-35           [-1, 1024, 2, 2]           2,048
                 ReLU-36           [-1, 1024, 2, 2]               0
               Conv2d-37           [-1, 1024, 2, 2]       9,438,208
          BatchNorm2d-38           [-1, 1024, 2, 2]           2,048
                 ReLU-39           [-1, 1024, 2, 2]               0
            MaxPool2d-40           [-1, 1024, 1, 1]               0
              Flatten-41                 [-1, 1024]               0
              Dropout-42                 [-1, 1024]               0
               Linear-43                  [-1, 512]         524,800
                 ReLU-44                  [-1, 512]               0
               Linear-45                   [-1, 11]           5,643
    ================================================================
    Total params: 29,891,467
    Trainable params: 29,891,467
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.19
    Forward/backward pass size (MB): 142.50
    Params size (MB): 114.03
    Estimated Total Size (MB): 256.71
    ----------------------------------------------------------------


    /Users/tonmoydey/opt/anaconda3/lib/python3.8/site-packages/torch/nn/functional.py:1639: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
      warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")


### **17 - Training the Model**


```python
#  Hyperparameters
epochs = 3
lr = 0.001
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
```


```python
# Training

history = fit(epochs, lr, model, train_dl, val_dl, grad_clip=grad_clip, weight_decay=weight_decay, opt_func=opt_func)
```


    HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=303.0), HTML(value='')))


    
    {'val_loss': 0.1452827751636505, 'val_score': 0.8190315365791321, 'train_loss': 0.21707668900489807}
    Epoch [0], train_loss: 0.2171, val_loss: 0.1453, val_score: 0.8190



    HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=303.0), HTML(value='')))


    
    {'val_loss': 0.10721645504236221, 'val_score': 0.8889080882072449, 'train_loss': 0.12184283137321472}
    Epoch [1], train_loss: 0.1218, val_loss: 0.1072, val_score: 0.8889



    HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=303.0), HTML(value='')))


    
    {'val_loss': 0.0938473716378212, 'val_score': 0.908584475517273, 'train_loss': 0.10529273748397827}
    Epoch [2], train_loss: 0.1053, val_loss: 0.0938, val_score: 0.9086


### **18 - Plotting the Metrics**


```python
def plot_scores(history):
  scores = [x['val_score'] for x in history]
  plt.plot(scores)
  plt.xlabel('epoch')
  plt.ylabel('score')
  plt.title('F1 Score Vs. Epoch')
  plt.show()

def plot_losses(history):
  val_losses = [x['val_loss'] for x in history]
  train_losses = [x['train_loss'] for x in history]
  plt.plot(val_losses, '-rx', label='Validation')
  plt.plot(train_losses, '-bx', label='Train')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend()
  plt.title('Loss Vs. Epoch')
  plt.show()
```


```python
plot_scores(history)
plot_losses(history)
```


    
![png](output_50_0.png)
    



    
![png](output_50_1.png)
    


### **19 - Testing the Model**


```python
def predict_single(image):
  x = image.unsqueeze(0)
  x = to_device(x, device)
  preds = model(x)
  prediction = preds[0]
#   print('Output Tensor: ', preds)
#   print('Prediction Tensor: ', prediction)
  print('Prediction Label: ', decode(prediction))

  show_image(image, prediction)

predict_single(val_ds[80][0])
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


    Prediction Label:  blue shirt
    Label:  blue shirt
    tensor([1.6171e-03, 9.9798e-01, 6.5156e-06, 3.2483e-03, 5.6052e-06, 3.7909e-07,
            2.8269e-04, 9.2780e-02, 1.2908e-02, 1.0480e-03, 9.0414e-01],
           grad_fn=<SelectBackward>)



    
![png](output_52_2.png)
    



```python

```


```python

```
