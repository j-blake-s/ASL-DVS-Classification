import numpy as np
import cupy as cp

def aslAugment(x):
  cx = cp.asarray(x)
  # cx = salt_and_pepper(cx, p_salt=0.001, p_pepper=0.3)
  # cx = temporal_jitter(cx, max_shift=5)
  # cx = spatial_jitter(cx, max_shift=10)
  x = cp.asnumpy(cx)
  return x

def salt_and_pepper(x, p_salt, p_pepper):
  C, H, W, T = x.shape
  x = x.reshape(C, -1, T)

  def _sop(p):
    ids = np.arange(H*W)
    np.random.shuffle(ids)
    return ids[:int(H*W*p)]
  
  x[:, _sop(p_pepper), :] = 0
  x[:, _sop(p_salt), :] = 1
  return x.reshape(C, H, W, T)

def temporal_jitter(image, max_shift=3):
  dt = np.random.randint(-max_shift, high=max_shift+1)
  temp = cp.zeros_like(image)
  if    dt < 0: temp[:,:,:,:dt] = image[:,:,:,-dt:]
  elif  dt > 0: temp[:,:,:,dt:] = image[:,:,:,:-dt]
  else: return image
  return temp

def spatial_jitter(image, max_shift=10):
  dh = np.random.randint(-max_shift, high=max_shift+1)
  dw = np.random.randint(-max_shift, high=max_shift+1)
  
  _, H, W, _ = image.shape
  temp = cp.zeros_like(image)

  def idxs(shift, max_idx):
    if shift >= 1: return  (shift, max_idx), (0, max_idx-shift)
    elif shift==0: return (0,max_idx),  (0,max_idx)
    else: return      (0,max_idx+shift),(-shift, max_idx)

  (ihl, ihr), (thl, thr) = idxs(dh, H)
  (iwl, iwr), (twl, twr)  = idxs(dw, W)
  
  temp[:,thl:thr,twl:twr,:] = image[:,ihl:ihr,iwl:iwr,:]
  return temp