import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(torch.nn.Module):
  def __init__(self, channels, classes):
    super().__init__()


    # Conv Layers
    self.convs = torch.nn.ModuleList([
      nn.Conv2d(2*groups, 16, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(64, 72, kernel_size=3, stride=1, padding=1),
    ])

    # Dense Layers
    self.dense = torch.nn.ModuleList([
      nn.Linear(2*3*72, 512),
      nn.Linear(512, 128),
    ])

    # Output layer
    self.output = nn.Linear(self.dense[-1].out_features, num_classes)
    self.dropout = nn.Dropout(0.3)


  def forward(self, x):

    x = self.acc(x)
    # x = self.normalizer(x)

    for conv in self.convs[:]: 
      x = conv(x)
      x = F.relu(x)
      x = F.max_pool2d(x, 2)

    x = torch.flatten(x, 1)

    for d in self.dense:
      x = d(x)
      x = F.relu(x)
      x = self.dropout(x)

    x = self.output(x)
    x = F.relu(x)
    x = F.softmax(x, dim=1)
    return x 



def load_model(args):
  model = ACNN().to(args.device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  error = torch.nn.CrossEntropyLoss().to(args.device)
  classer = lambda x: torch.argmax(x,axis=-1)
  return model, optimizer, error, classer