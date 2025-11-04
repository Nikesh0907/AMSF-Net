import os
import numpy as np

def _load_npy(root, name):
    p = os.path.join(root, name)
    if os.path.exists(p):
        return np.load(p, allow_pickle=True)
    return None

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
