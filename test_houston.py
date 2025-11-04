

import os
import numpy as np
import scipy.io as sio

from models.AMSF import AMSF
from utils import *
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam, calc_cc, calc_moae, calc_uiqi, calc_ssim
import args_parser
from time import *
from build_datasets_houston import *

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(args)

def main():

    args.n_bands = 46

    _, _, test_ref, test_lr, test_hr = all_train_test_data_in()
    # Build the models

    model = AMSF(
                     args.scale_ratio,
                     args.n_select_bands,
                     args.n_bands).cuda()


    # Load the trained model parameters
    # Use the same model_path convention as main.py
    model_path = args.model_path.replace('dataset', args.dataset)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print('Loaded checkpoint: {}'.format(model_path))
    else:
        print('Warning: checkpoint not found at {}. Proceeding with randomly initialized weights.'.format(model_path))


    # test_ref, test_lr, test_hr = test_list
    test_ref = torch.Tensor(test_ref)
    test_lr = torch.Tensor(test_lr)
    test_hr = torch.Tensor(test_hr)

    test_ref = test_ref.permute(0, 3, 1, 2)
    test_lr = test_lr.permute(0, 3, 1, 2)
    test_hr = test_hr.permute(0, 3, 1, 2)

    model.eval()

    # Set mini-batch dataset
    ref = test_ref.float().detach()
    lr = test_lr.float().detach()
    hr = test_hr.float().detach()
    model.cuda()
    lr = lr.cuda()
    hr = hr.cuda()

    out,_,_,_ = model(lr, hr)




    # flops
    print()
    print()

    ref = ref.detach().cpu().numpy()
    out = out.detach().cpu().numpy()

    os.makedirs('./result', exist_ok=True)
    N = out.shape[0]
    PSNR = AverageMeter()
    for i in range(N):
        ref_i = ref[i]
        out_i = out[i]
        # save as HxWxC .mat for inspection
        sio.savemat('./result/{}_{}_ref.mat'.format(i, args.dataset), {'ref': np.transpose(ref_i, (1, 2, 0))})
        sio.savemat('./result/{}_{}_out.mat'.format(i, args.dataset), {'out': np.transpose(out_i, (1, 2, 0))})
        try:
            psnr_i = calc_psnr(ref_i, out_i)
            PSNR.update(psnr_i, 1)
        except Exception:
            pass
    print('Avg PSNR: {:.4f}'.format(PSNR.avg if PSNR.count>0 else 0.0))




if __name__ == '__main__':
    main()
