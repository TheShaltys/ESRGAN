import sys
import os.path
import glob
import cv2
import numpy as np
import torch
import architecture as arch
from datetime import datetime, date

model_path = sys.argv[1]  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
output_file = sys.argv[2]
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'ESRGAN/LR/*'

model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))
print('Frames: ', len(glob.glob(test_img_folder)))

dt_start = datetime.strptime('07:50:51', '%H:%M:%S').time()
start_date = date.today()

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = os.path.splitext(os.path.basename(path))[0]
    now = datetime.now()
    dt_string = now.strftime("%H:%M:%S")
    
    print(idx, base, dt_string)
    # read image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(output_file.format(base), output)
 
dt_end = datetime.now().time()
end_date = date.today()

combination_start = datetime.combine(start_date, dt_start)
combination_end = datetime.combine(end_date, dt_end)

duration = datetime.strptime(str(combination_end - combination_start), '%H:%M:%S.%f')

print('Duration: ',duration.strftime('%H:%M:%S'), ', Frames: ', len(glob.glob(test_img_folder)))
