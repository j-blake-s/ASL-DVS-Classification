import torch
def train(model, data, opt, err, pred, args):
  model.train()
  total_samples = 0
  loss_sum = 0
  total_correct_samples = 0
  for i, (x, y) in enumerate(data):
    # Prepare Data #
    x, y = x.to(args.device), y.to(args.device)
    x = x.to(torch.float32)

    # Run Model #
    outputs = model(x)

    # Calculate Loss #
    loss = err(outputs, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    
    # Update Stats #
    total_samples += x.shape[0]
    loss_sum += loss.cpu().data.item() * y.shape[0]
    correct_samples = torch.sum(pred(outputs)==y).cpu().data.item()
    total_correct_samples += correct_samples
    acc = total_correct_samples / total_samples
    print(f'\r\tBatch [{i+1}/{len(data)}] Training: {acc:.2%}',end="")
  return acc

def test(model, data, pred, args):
  model.eval()
  total_samples = 0
  total_correct_samples = 0
  with torch.no_grad():
    for i, (x, y) in enumerate(data):
      # Prepare Data #
      x, y = x.to(args.device), y.to(args.device)
      x = x.to(torch.float32)

      # Run Model #
      outputs = model(x)

      # Update Stats #
      total_samples += x.shape[0]
      correct_samples = torch.sum(pred(outputs)==y).cpu().data.item()
      total_correct_samples += correct_samples
      acc = total_correct_samples / total_samples
      print(f'\r\tBatch [{i+1}/{len(data)}] Validation: {acc:.2%}',end="")
  return acc