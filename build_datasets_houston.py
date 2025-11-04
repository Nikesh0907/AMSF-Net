import os
import numpy as np
import os.path as osp
from typing import Tuple, Optional
try:
  import scipy.io as sio
except Exception:
  sio = None

def _load_npy(root, name):
    p = os.path.join(root, name)
    if os.path.exists(p):
        return np.load(p, allow_pickle=True)
    return None

def _maybe_expand_3d(x: np.ndarray) -> np.ndarray:
  # Ensure (N,H,W,C) by adding batch dim if only (H,W,C)
  if x is None:
    return x
  if x.ndim == 3:
    x = x[None, ...]
  return x


def _load_from_mat(mat_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
  """
  Attempt to load expected arrays from a .mat file with keys (case-insensitive):
    train_ref, train_hr, test_ref, test_lr, test_hr
  Returns tuple or None if not all keys are found.
  """
  if sio is None:
    return None
  if not osp.isfile(mat_path):
    return None
  try:
    m = sio.loadmat(mat_path)
  except Exception:
    return None
  keys = {k.lower(): k for k in m.keys() if not k.startswith('_')}
  want = ['train_ref', 'train_hr', 'test_ref', 'test_lr', 'test_hr']
  if not all(w in keys for w in want):
    # Try a few synonyms
    synonyms = {
      'train_ref': ('trainref', 'tr_ref', 'trref', 'train_hsi', 'tr_hsi'),
      'train_hr': ('trainhr', 'tr_hr', 'trhr', 'train_msi', 'tr_msi'),
      'test_ref': ('testref', 'te_ref', 'teref', 'test_hsi', 'te_hsi'),
      'test_lr': ('testlr', 'te_lr', 'telr', 'test_msi', 'te_msi'),
      'test_hr': ('testhr', 'te_hr', 'tehr', 'test_msi_hr', 'te_msi_hr'),
    }
    resolved = {}
    for w in want:
      if w in keys:
        resolved[w] = m[keys[w]]
        continue
      for alt in synonyms.get(w, ()): 
        if alt in keys:
          resolved[w] = m[keys[alt]]
          break
    if len(resolved) != len(want):
      # Not enough signals; print available keys for guidance
      avail = [k for k in m.keys() if not k.startswith('_')]
      print('[build_datasets_houston] Could not find all arrays in .mat. Available keys:', avail)
      return None
    train_ref = resolved['train_ref']
    train_hr  = resolved['train_hr']
    test_ref  = resolved['test_ref']
    test_lr   = resolved['test_lr']
    test_hr   = resolved['test_hr']
  else:
    train_ref = m[keys['train_ref']]
    train_hr  = m[keys['train_hr']]
    test_ref  = m[keys['test_ref']]
    test_lr   = m[keys['test_lr']]
    test_hr   = m[keys['test_hr']]

  # Ensure dtype float32 and minimum 4D with batch dim
  arrs = []
  for a in (train_ref, train_hr, test_ref, test_lr, test_hr):
    a = np.array(a)
    a = a.astype('float32', copy=False)
    a = _maybe_expand_3d(a)
    arrs.append(a)
  return tuple(arrs)


def all_train_test_data_in(root=None):
    """
    Minimal dataset loader used for smoke tests.
    It attempts to load these files from `root`:
      - train_ref.npy, train_hr.npy, test_ref.npy, test_lr.npy, test_hr.npy
    If files are not present it returns small random arrays suitable for a quick sanity check.

    Return order expected by the repo:
      train_ref, train_hr, test_ref, test_lr, test_hr
    Shapes: (N, H, W, C)
    """
  # Resolve root path: use CLI --root if available
  if root is None:
    try:
      import args_parser
      _args = args_parser.args_parser()
      base = getattr(_args, 'root', './data')
    except Exception:
      base = './data'
    # prefer a subfolder named 'houston'
    if os.path.basename(base).lower() == 'houston':
      root = base
    else:
      root = os.path.join(base, 'houston')

  # If root is a .mat file, try to load from it directly
  if isinstance(root, str) and root.lower().endswith('.mat') and osp.isfile(root):
    loaded = _load_from_mat(root)
    if loaded is not None:
      return loaded
    # Fall through to directory-based loading if .mat parsing failed

  train_ref = _load_npy(root, 'train_ref.npy')
    train_hr = _load_npy(root, 'train_hr.npy')
    test_ref = _load_npy(root, 'test_ref.npy')
    test_lr = _load_npy(root, 'test_lr.npy')
    test_hr = _load_npy(root, 'test_hr.npy')

    if test_ref is None or test_lr is None or test_hr is None:
        # Create tiny random arrays for smoke-testing
        N_test = 2
        H, W = 64, 64
        n_bands = 46
        test_ref = np.random.rand(N_test, H, W, n_bands).astype('float32')
        # test_lr and test_hr are commonly MSI with 3 channels (RGB-like), adjust if your data differs
        test_lr = np.random.rand(N_test, H//4, W//4, 3).astype('float32')
        test_hr = np.random.rand(N_test, H, W, 3).astype('float32')
        train_ref = np.empty((0,))
        train_hr = np.empty((0,))

    return train_ref, train_hr, test_ref, test_lr, test_hr
