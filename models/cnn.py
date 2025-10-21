import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(torch.nn.Module):
  def __init__(self, args):
    super().__init__()


    # Layers
    self.net = nn.Sequential(
      nn.Conv3d(args.channels, 4, kernel_size=(3,3,3), stride=1, padding=1),
      nn.ReLU(),
      nn.AvgPool3d(kernel_size=(2,2,2), stride=(2,2,2)),

      nn.Conv3d(4, 8, kernel_size=(3,3,3), stride=1, padding=1),
      nn.ReLU(), 
      nn.AvgPool3d(kernel_size=(2,2,2), stride=(2,2,2)),

      nn.Conv3d(8, 16, kernel_size=(3,3,3), stride=1, padding=1),
      nn.ReLU(),
      nn.AvgPool3d(kernel_size=(2,2,2), stride=(2,2,2)),
    
      nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=1, padding=1),
      nn.ReLU(),
      nn.AvgPool3d(kernel_size=(2,2,2), stride=(2,2,2)),    

      nn.Conv3d(32, 64, kernel_size=(3,3,1), stride=1, padding=(1,1,0)),
      nn.ReLU(),
      nn.AvgPool3d(kernel_size=(2,2,1), stride=(2,2,1)),   

      nn.Conv3d(64, 64, kernel_size=(3,3,1), stride=1, padding=(1,1,0)),
      nn.ReLU(),
      nn.AvgPool3d(kernel_size=(2,2,1), stride=(2,2,1)),   


    )


    # Dense Layers
    self.dense = torch.nn.ModuleList([
      nn.Linear(64*30, 256),
      nn.Linear(256, 64),
    ])

    # Output layer
    self.output = nn.Linear(self.dense[-1].out_features, args.classes)
    self.dropout = nn.Dropout(0.2)


  def forward(self, x):

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
  model = CNN(args).to(args.device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  error = torch.nn.CrossEntropyLoss().to(args.device)
  classer = lambda x: torch.argmax(x,axis=-1)
  return model, optimizer, error, classer