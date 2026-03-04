import torch
import lava.lib.dl.slayer as slayer
import torch.nn.functional as F

neuron_params = {
  'threshold'     : 1.25,
  'current_decay' : 0.25,
  'voltage_decay' : 0.03,
  'tau_grad'      : 0.03,
  'requires_grad' : True,
  'dropout'       : slayer.neuron.Dropout(p=0.05),
}

def SlayerConv(in_, out_, kernel_size=3, stride=1, padding=1):
  return slayer.block.cuba.Conv(
    neuron_params, in_, out_, kernel_size=kernel_size, stride=stride, 
    padding=padding, weight_scale=2, weight_norm=True, delay=False, 
    delay_shift=False,
  )

def SlayerPool(size=2):
  return slayer.block.cuba.Pool(
    neuron_params, kernel_size=size, stride=size, delay=False, delay_shift=False
  )

def SlayerDense(in_, out_):
  return slayer.block.cuba.Dense(
    neuron_params, in_, out_, weight_scale=2, weight_norm=True, delay_shift=False,
  )


class MyConv(torch.nn.Module):
  def __init__(self, in_, out_):
    super().__init__()

    self.conv = torch.nn.Conv2d(in_,out_, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    B, C, H, W, T = x.shape

    x = torch.permute(x, (4, 0, 1, 2, 3)).flatten(0,1)
    x = self.conv(x)
    x = F.relu(x)

    _, C, H, W = x.shape
    x = x.reshape(T, B, C, H, W)
    return torch.permute(x, (1, 2, 3, 4, 0))
  

class MyPool(torch.nn.Module):
  def __init__(self, size):
    super().__init__()

    self.pool = torch.nn.MaxPool2d(size)

  def forward(self, x):
    B, C, H, W, T = x.shape

    x = torch.permute(x, (4, 0, 1, 2, 3)).flatten(0,1)
    x = self.pool(x)

    _, C, H, W = x.shape
    x = x.reshape(T, B, C, H, W)
    return torch.permute(x, (1, 2, 3, 4, 0))
  
  
class MyDense(torch.nn.Module):
  def __init__(self, in_, out_):
    super().__init__()

    self.lin = torch.nn.Linear(in_,out_)

  def forward(self, x):
    B, C, T = x.shape

    x = torch.permute(x, (2, 0, 1)).flatten(0,1)
    x = self.lin(x)
    _, C = x.shape
    x = x.reshape(T, B, C)
    x = torch.permute(x, (1, 2, 0))
    return x
  
class DeepSNN(torch.nn.Module):

  ### Init ###
  def __init__(self, args):
    super(DeepSNN, self).__init__()

    SpikeConv = SlayerConv
    SpikePool = SlayerPool
    SpikeDense = SlayerDense

    self.conv = torch.nn.ModuleList([
      SpikeConv(2, 4),
      MyConv(4, 6),
      MyConv(6, 12),
      MyConv(12, 24),
      MyConv(24, 32),
    ])

    self.pool = torch.nn.ModuleList([
      MyPool(4),
      MyPool(2),
      MyPool(2),
      MyPool(2),
      MyPool(2),
    ]) 

    self.flatten = slayer.block.cuba.Flatten()

    self.dense = torch.nn.ModuleList([
      MyDense(32*5*3, 256),
      torch.nn.Dropout(0.15),
      torch.nn.ReLU(),

      MyDense(256, 64),
      torch.nn.Dropout(0.15),
      torch.nn.ReLU(),

      MyDense(64, args.classes),
    ])


  ### Forward Pass ###
  def forward(self, x):
    for c,p in zip(self.conv,self.pool):
      x = c(x)
      # print("Post Conv", x.shape)
      x = p(x)
      # print("Post Pool", x.shape)

    x = self.flatten(x)

    for d in self.dense:
      x = d(x)

    return x

  
def load_model(args):
  model = DeepSNN(args).to(args.device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  error = slayer.loss.SpikeRate(true_rate=0.8, false_rate=0.2, reduction='sum').to(args.device)
  classer = slayer.classifier.Rate.predict

  return model, optimizer, error, classer
