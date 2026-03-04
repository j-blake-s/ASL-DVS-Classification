import torch
from torch import nn
# from .spikingV2 import SpkConv, SpkDense
import lava.lib.dl.slayer as slayer
import torch.nn.functional as F



class MaxPool(nn.Module):
  def __init__(self, kernel_size=2, stride=2):
    super().__init__()

    self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

  def forward(self, x):
    B, C, H, W, T  = x.shape
    x = torch.permute(x, (4, 0, 1, 2, 3))
    x = x.flatten(0,1)
    x = self.pool(x)
    _, _, H, W = x.shape
    x = x.reshape(T, B, C, H, W)
    x = torch.permute(x, (1, 2, 3, 4, 0))
    return x

class SNN(nn.Module):
  def __init__(self, args):
    super().__init__()


    pool = nn.MaxPool2d
    self.net = nn.Sequential(

      nn.Conv2d(2, 6, 3),
      pool(4),
      nn.ReLU(),

      nn.Conv2d(6, 16, 3),
      pool(4),
      nn.ReLU(),

      nn.Conv2d(16, 32, 3),
      pool(2),
      nn.ReLU(),


      nn.Conv2d(32, 64, 3),
      pool(2),
      nn.ReLU(),


      nn.Flatten(),
      
      nn.Linear(3*2*64, 256),
      nn.ReLU(),
      nn.Dropout(0.3),

      nn.Linear(256, args.classes),
      nn.ReLU(),

    ).to(args.device)

    # self.head = nn.Linear(45, 15)

  def params(self):
    model_ps = filter(lambda p: p.requires_grad, self.parameters())
    p_count = sum([torch.prod(torch.tensor(p.size())) for p in model_ps])
    return p_count.item()
  
  def forward(self, x):
    B, C, H, W, T = x.shape

    x = torch.permute(x, (4, 0, 1, 2, 3)).flatten(0,1) # TB C H w
    x = self.net(x).reshape(T, B, -1)
    x = torch.permute(x, (1, 2, 0))
    # x = self.head(x).squeeze() # B C 1
    # x = torch.sum(x, dim=-1)
    # x = F.softmax(x, dim=-1)
    return x
  


def load_model(args):
  model = SNN(args).to(args.device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  # error = torch.nn.CrossEntropyLoss().to(args.device)
  # classer = lambda x: torch.argmax(x,axis=-1)
  error = slayer.loss.SpikeRate(true_rate=0.8, false_rate=0.2, reduction='mean').to(args.device)
  classer = slayer.classifier.Rate.predict
  return model, optimizer, error, classer




