import cv2
import getopt
import numpy as np
import sys
import torch

from models.common import Concat, Conv, MP, RepConv, SPPCSPC
from models.experimental import attempt_load
from models.yolo import Detect, IDetect, IKeypoint


outputs = {}
def get_output(name):
    def hook(model, input, output):
        outputs[name] = (output.shape, output.detach())
    return hook


def main(input_filename, model_filename, input_size):
    device = 'cpu'
    model = attempt_load(model_filename, map_location=device)  # load FP32 model
    image = cv2.imread(input_filename)  # BGR
    image = cv2.resize(image, (input_size, input_size))
    image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).to(device)
    image = image.float()
    image /= 255.0
    if image.ndimension() == 3:
        image = image.unsqueeze(0)

    
    model_children = list(model.children())
    counter = 0
    for l in model_children[0]:
        if isinstance(l, Conv):
            l.conv.register_forward_hook(get_output(str(counter)))
            l.act.register_forward_hook(get_output(str(counter + 1)))
            counter += 2
        elif isinstance (l, Concat):
            l.register_forward_hook(get_output(str(counter)))
            counter += 1
        elif isinstance (l, MP):
            l.m.register_forward_hook(get_output(str(counter)))
            counter += 1
        elif isinstance (l, SPPCSPC):
            l.cv2.conv.register_forward_hook(get_output(str(counter)))
            l.cv2.act.register_forward_hook(get_output(str(counter + 1)))
            l.cv1.conv.register_forward_hook(get_output(str(counter + 2)))
            l.cv1.act.register_forward_hook(get_output(str(counter + 3)))
            l.cv3.conv.register_forward_hook(get_output(str(counter + 4)))
            l.cv3.act.register_forward_hook(get_output(str(counter + 5)))
            l.cv4.conv.register_forward_hook(get_output(str(counter + 6)))
            l.cv4.act.register_forward_hook(get_output(str(counter + 7)))
            l.cv5.conv.register_forward_hook(get_output(str(counter + 8)))
            l.cv5.act.register_forward_hook(get_output(str(counter + 9)))
            l.cv6.conv.register_forward_hook(get_output(str(counter + 10)))
            l.cv6.act.register_forward_hook(get_output(str(counter + 11)))
            l.cv7.conv.register_forward_hook(get_output(str(counter + 12)))
            l.cv7.act.register_forward_hook(get_output(str(counter + 13)))
            counter += 14
        elif isinstance(l, RepConv):
            l.rbr_reparam.register_forward_hook(get_output(str(counter)))
            l.act.register_forward_hook(get_output(str(counter + 1)))
            counter += 2
        elif isinstance (l, torch.nn.Upsample):
            l.register_forward_hook(get_output(str(counter)))
            counter += 1
        elif isinstance(l, Detect):
            print('Detect')
        elif isinstance(l, IDetect):
            print('IDetect')
        elif isinstance(l, IKeypoint):
            for i in range(l.nl):
                l.m[i].register_forward_hook(get_output(str(counter)))
                for k in l.m_kpt[i]:
                    if isinstance(k, Conv):
                        k.conv.register_forward_hook(get_output(str(counter)))
                        k.act.register_forward_hook(get_output(str(counter + 1)))
                        counter += 2
                    elif isinstance(k, torch.nn.Conv2d):
                        k.register_forward_hook(get_output(str(counter)))
                        counter += 1

    pred = model(image)[0]
    l = [module for module in model.model.modules() if (not isinstance(module, torch.nn.Sequential) or not isinstance(module, Conv))]
    print(l)
    out = l[3].output




def show_usage():
    print('<debug.py> '
          '-i/--input_filename input filename | '
          '-m/--model_filename model filename | '
          '-s/--input_size input size | '
          '-h/--help')


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:m:s:', ['help',
                                                           'input_filename=',
                                                           'model_filename=',
                                                           'input_size='])
    except getopt.GetoptError as err:
        print(str(err))
        show_usage()
        sys.exit(2)

    input_filename = None
    model_filename = None
    input_size = 640

    for o, a in opts:
        if o in ('-h', '--help'):
            show_usage()
            sys.exit()
        elif o in ('-i', '--input_filename'):
            input_filename = a
        elif o in ('-m', '--model_filename'):
            model_filename = a
        elif o in ('-m', '--input_size'):
            input_size = int(a)

    main(input_filename, model_filename, input_size)
