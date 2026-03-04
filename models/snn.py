import torch
from torch import nn
# from .spikingV2 import SpkConv, SpkDense
import lava.lib.dl.slayer as slayer

params = {
  'threshold'     : 0.1,
  'current_decay' : 0.3,
  'voltage_decay' : 0.25,
  'tau_grad'      : 0.001,
  'requires_grad' : True,
}


from spikingjelly.clock_driven.neuron import MultiStepLIFNode as LIF

import snntorch as snn
from snntorch import surrogate
from snntorch import utils
from snntorch import functional as SF

class STorchDense(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.lin = nn.Linear(in_channels, out_channels)
    self.bn = nn.BatchNorm1d(out_channels)
    self.lif = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True)

  def forward(self, x):
    T, B, _,= x.shape
    x = self.lin(x)
    return x
    # x = self.bn(x)
    return self.lif(x)

class STorchConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    super().__init__()

    self.conv = nn.Conv2d(in_channels, out_channels, 
                          kernel_size=kernel_size, stride=stride, 
                          padding=padding, bias=False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.lif = snn.Leaky(beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True)

  def forward(self, x):
    T, B, _, _, _ = x.shape
    x = self.conv(x.flatten(0,1))
    
    _, C, H, W = x.shape
    x = self.bn(x).reshape(T, B, C, H, W)
    return x
    return self.lif(x)
  
class JellyDense(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
    self.bn = nn.BatchNorm1d(out_channels)
    self.lif = LIF(tau=2.0, detach_reset=True, backend='cupy')

  def forward(self, x):
    T, B, _,= x.shape
    x = self.conv(x)
    x = self.bn(x)
    return self.lif(x.contiguous())
class JellyConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
    super().__init__()

    self.conv = nn.Conv2d(in_channels, out_channels, 
                          kernel_size=kernel_size, stride=stride, 
                          padding=padding, bias=False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.lif = LIF(tau=2.0, detach_reset=True, backend='cupy')

  def forward(self, x):
    T, B, _, _, _ = x.shape
    x = self.conv(x.flatten(0,1))
    _, C, H, W = x.shape
    x = self.bn(x).reshape(T, B, C, H, W)
    return self.lif(x.contiguous())
def SlayerConv(in_, out_, kernel_size=3, stride=2, padding=1):
  return slayer.block.cuba.Conv(
    params, in_, out_, kernel_size=kernel_size, stride=stride, 
    padding=padding, weight_scale=2, weight_norm=True, delay=False, 
    delay_shift=False,
  )
def SlayerDense(in_, out_):
  return slayer.block.cuba.Dense(
    params, in_, out_, weight_scale=2, weight_norm=True, delay_shift=False,
  )


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

    beta = 0.5
    grad = snn.surrogate.fast_sigmoid(slope=25)

    self.net = nn.Sequential(

      nn.Conv2d(2, 6, 3), # in_channels, out_channels, kernel_size
      nn.MaxPool2d(4),
      snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
      # nn.ReLU(),

      nn.Conv2d(6, 16, 3),
      nn.MaxPool2d(4),
      snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
      # nn.ReLU(),

      nn.Conv2d(16, 32, 3),
      nn.MaxPool2d(2),
      snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
      # nn.ReLU(),


      nn.Conv2d(32, 64, 3),
      nn.MaxPool2d(2),
      snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
      # nn.ReLU(),


      nn.Flatten(),
      
      nn.Linear(3*2*64, 256),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(256, args.classes),
      nn.ReLU(),
    ).to(args.device)

    # Dense = SlayerDense
    # Conv = SlayerConv
    # self.conv = nn.ModuleList([
    #   Conv(args.channels, 8),
    #   MaxPool(2,2),

    #   Conv(8, 32),
    #   MaxPool(2,2),

    #   Conv(32, 64),
    #   MaxPool(2,2),

    # ])

    # self.dense = nn.ModuleList([
    #   Dense(5*4*64, 256),
    #   # nn.Dropout(0.2),
    #   Dense(256, 64),
    #   # nn.Dropout(0.2),
    #   Dense(64, args.classes),
    # ])


  def params(self):
    model_ps = filter(lambda p: p.requires_grad, self.parameters())
    p_count = sum([torch.prod(torch.tensor(p.size())) for p in model_ps])
    return p_count.item()
  
  def forward(self, x):
    B, C, H, W, T = x.shape

    spk_rec = []

    x = torch.permute(x, (4, 0, 1, 2, 3))
    snn.utils.reset(self.net)

    for data_t in x:
      out = self.net(data_t)
      spk_rec.append(out)

    out = torch.stack(spk_rec)

    return torch.permute(out, (1, 2, 0))
    # x = torch.permute(x, (4, 0, 1, 2, 3))
    
    # # Conv
    # for c in self.conv:
    #   x = c(x)

    # # Flatten
    # x = x.flatten(1,3)

    # # Dense
    # for d in self.dense:
    #   x = d(x)

    # x = torch.permute(x, (1,2,0))

    # return out
  


def load_model(args):
  model = SNN(args).to(args.device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  error = slayer.loss.SpikeRate(true_rate=0.8, false_rate=0.2, reduction='mean').to(args.device)
  # error = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
  classer = slayer.classifier.Rate.predict
  return model, optimizer, error, classer




