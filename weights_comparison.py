import getopt
import numpy as np
from os import makedirs, path
from shutil import copyfile
import sys
import torch
import torch.nn as nn

from models.common import Conv, RepConv, SPPCSPC
from models.yolo import Detect, IDetect
from models.experimental import attempt_download, attempt_load, Ensemble
from utils.torch_utils import select_device


def load_model(model_filename, device):
    model = Ensemble()
    for w in weights if isinstance(model_filename, list) else [model_filename]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=device)
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())
 
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()
 
    if len(model) == 1:
        model =  model[-1]
    else:
        print('Ensemble created with %s\\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
     
    return model


def main(model1_filename, model2_filename):
    device = select_device('cpu', batch_size=1)
    
    model1 = load_model(model1_filename, device)
    model2 = load_model(model2_filename, device)
            
    models_differ = 0
    for key_item_1, key_item_2 in zip(model1.state_dict().items(), model2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
        
    if models_differ == 0:
        print('Models match perfectly! :)')


def show_usage():
    print('<weights_comparison.py> '
          '--m1 model1 filename | '
          '--m2 model2 filename | '
          '-h/--help')


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'h', ['help',
                                                       'm1=',
                                                       'm2='])
    except getopt.GetoptError as err:
        print(str(err))
        show_usage()
        sys.exit(2)

    model1_filename = None
    model2_filename = None

    for o, a in opts:
        if o in ('-h', '--help'):
            show_usage()
            sys.exit()
        elif o in ('--m1'):
            model1_filename = a
        elif o in ('--m2'):
            model2_filename = a

    main(model1_filename, model2_filename)
