import torch
import torch.nn as nn
import torch.nn.functional as F

class Accumulate(nn.Module):
  def __init__(self, T, I):
    super().__init__()
    self.timesteps = T
    self.interval = I

    self.conv = nn.Conv3d(1,1,kernel_size=(1,1,I), stride=(1,1,I), padding=(0,0,I//2), bias=False)
    self.conv.requires_grad = False
    with torch.no_grad():
      self.conv.weight.data = torch.ones(size=(1,1,1,1,I))

  def forward(self, x):
    B, C, H, W, T = x.shape
    x = x.reshape(B*C, 1, H, W, T)
    x = self.conv(x) # B*C 1 H W T/I
    x = torch.permute(x, (0, 1, 4, 2, 3)).reshape(B, -1, H, W)
    return x


class ACNN(torch.nn.Module):
  def __init__(self, args):
    super().__init__()

    self.accumulate = Accumulate(args.timesteps, args.interval)

    # Layers
    self.net = nn.Sequential(
      nn.Conv2d(args.channels*args.timesteps // args.interval, 4, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.AvgPool2d(kernel_size=2, stride=2),

      nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
      nn.ReLU(), 
      nn.AvgPool2d(kernel_size=2, stride=2),

      nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.AvgPool2d(kernel_size=2, stride=2),
    
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.AvgPool2d(kernel_size=2, stride=2),    

      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.AvgPool2d(kernel_size=2, stride=2),   

      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.AvgPool2d(kernel_size=2, stride=2),   


    )


    # Dense Layers
    self.dense = torch.nn.ModuleList([
      nn.Linear(5*3*64, 256),
      nn.Linear(256, 64),
    ])

    # Output layer
    self.output = nn.Linear(self.dense[-1].out_features, args.classes // 2 if args.combine_classes else args.classes)
    self.dropout = nn.Dropout(0.2)


  def forward(self, x):
    
    x = self.accumulate(x)

    x = self.net(x)

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
  model = ACNN(args).to(args.device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  error = torch.nn.CrossEntropyLoss().to(args.device)
  classer = lambda x: torch.argmax(x,axis=-1)
  return model, optimizer, error, classer