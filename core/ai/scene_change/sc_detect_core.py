import base64

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from core.ai.scene_change.model.TANet import TANet

""" ------------------------------------------------------------------------------------ """
""" MODEL INIT                                                                           """
""" ------------------------------------------------------------------------------------ """
model_path = Config.model_path

model = TANet(encoder_arch="resnet18", local_kernel_size=1, stride=1, padding=0, groups=4, drtam=True,
              refinement=True)

if torch.cuda.is_available():
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.cpu()
    model.eval()


def scene_change_detect(old_image, new_image):
    """ ------------------------------------------------------------------------------------ """
    """ PREPROCESS IMAGES                                                                    """
    """ ------------------------------------------------------------------------------------ """
    w, h, c = old_image.shape
    w_r = int(256 * max(w / 256, 1))
    h_r = int(256 * max(h / 256, 1))

    # resize images so that min(w, h) == 256
    img_t0_r = cv2.resize(old_image, (h_r, w_r))
    img_t1_r = cv2.resize(new_image, (h_r, w_r))

    img_t0_r = np.asarray(img_t0_r).astype('f').transpose(2, 0, 1) / 128.0 - 1.0
    img_t1_r = np.asarray(img_t1_r).astype('f').transpose(2, 0, 1) / 128.0 - 1.0

    input = torch.from_numpy(np.concatenate((img_t0_r, img_t1_r), axis=0)).contiguous()
    input = input.view(1, -1, w_r, h_r)

    if torch.cuda.is_available():
        input = input.cuda()
    else:
        input = input.cpu()

    output = model(input)

    """ ------------------------------------------------------------------------------------ """
    """ RESULT                                                                               """
    """ ------------------------------------------------------------------------------------ """
    output = output[0].cpu().data

    mask_pred = np.where(F.softmax(output[0:2, :, :], dim=0)[0] > 0.5, 255, 0)
    mask_pred = cv2.cvtColor(mask_pred.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    retval, buffer = cv2.imencode('.jpg', mask_pred)

    pred_b64 = base64.b64encode(buffer).decode("utf-8")

    return pred_b64
