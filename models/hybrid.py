import torch
from torch import nn
# from .spikingV2 import SpkConv, SpkDense
import lava.lib.dl.slayer as slayer

# from spikingjelly.clock_driven.neuron import MultiStepLIFNode as LIF
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



params = {
  'threshold'     : 1,
  'current_decay' : 0.3,
  'voltage_decay' : 0.25,
  'tau_grad'      : 0.001,
  'requires_grad' : True,
}

def SpkConv(in_, out_, kernel_size=3, stride=1, padding=1):
  return slayer.block.cuba.Conv(
    params, in_, out_, kernel_size=kernel_size, stride=stride, 
    padding=padding, weight_scale=2, weight_norm=True, delay=False, 
    delay_shift=False,
  )
def SpkDense(in_, out_):
  return slayer.block.cuba.Dense(
    params, in_, out_, weight_scale=2, weight_norm=True, delay_shift=False,
  )

class Hybrid(nn.Module):
  def __init__(self, args):
    super().__init__()

    self.spiking = nn.ModuleList([
      SpkConv(args.channels,2),
      Accumulate(args.timesteps, args.interval),
      nn.Conv2d(2 * args.timesteps // args.interval, 4, kernel_size=1, stride=1),
      nn.ReLU(),
      nn.AvgPool2d(kernel_size=2, stride=2)
    ])

    self.conv = nn.ModuleList([
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
    ])


    # Dense Layers
    self.dense = torch.nn.ModuleList([
      nn.Linear(5*3*64, 256),
      nn.Linear(256, 64),
    ])

    # Output layer
    self.output = nn.Linear(self.dense[-1].out_features, args.classes)
    self.dropout = nn.Dropout(0.2)


  def params(self):
    model_ps = filter(lambda p: p.requires_grad, self.parameters())
    p_count = sum([torch.prod(torch.tensor(p.size())) for p in model_ps])
    return p_count.item()
  
  def forward(self, x):
    B, C, H, W, T = x.shape


    # Spiking 
    for s in self.spiking:
      x = s(x)

    # Conv
    for c in self.conv:
      x = c(x)

    x = torch.flatten(x, 1)

    for d in self.dense:
      x = d(x)
      x = F.relu(x)
      x = self.dropout(x)

    x = self.output(x)
    x = F.relu(x)
    x = F.softmax(x, dim=1)
    return x 
  

# def load_model(args):
#   model = SNN(args).to(args.device)
#   optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#   error = slayer.loss.SpikeRate(true_rate=0.5, false_rate=0.05, reduction='mean').to(args.device)
#   classer = slayer.classifier.Rate.predict
#   return model, optimizer, error, classer

def load_model(args):
  model = Hybrid(args).to(args.device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  error = torch.nn.CrossEntropyLoss().to(args.device)
  classer = lambda x: torch.argmax(x,axis=-1)
  return model, optimizer, error, classer