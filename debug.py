import cv2
import getopt
import numpy as np
import sys
import torch

from models.common import Concat, Conv, MP, RepConv, SPPCSPC
from models.experimental import attempt_load
from models.yolo import Detect, IDetect, IKeypoint


class SaveOutput:
    def __init__(self):
        self.outputs = []
 
    def __call__(self, module, module_in, module_out):
        layer_type = None
        if isinstance(module, torch.nn.Conv2d):
            layer_type = 'conv2d  '
        elif isinstance(module, torch.nn.ReLU):
            layer_type = 'relu    '
        elif isinstance(module, torch.nn.SiLU):
            layer_type = 'silu    '
        elif isinstance(module, torch.nn.Identity):
            layer_type = 'identity'
        elif isinstance (module, Concat):
            layer_type = 'concat  '
        elif isinstance (module, torch.nn.MaxPool2d):
            layer_type = 'maxpool '
        elif isinstance (module, torch.nn.Upsample):
            layer_type = 'upsample'
        else:
            print('layer type not found')

        if isinstance(module_in[0], list) == False:
            self.outputs.append((layer_type,
                                module_in[0].shape,
                                module_in[0].detach(),
                                module_out.shape,
                                module_out.detach()))
        else:
            self.outputs.append((layer_type,
                                module_out.shape,
                                module_out.detach(),
                                module_out.shape,
                                module_out.detach()))
        if isinstance(module, torch.nn.Conv2d):
            self.outputs[-1] += (module.weight.shape,
                                 module.weight.detach(),
                                 module.bias.shape,
                                 module.bias.detach())
 
    def clear(self):
        self.outputs = []


def main(input_filename, model_filename, input_size, output_filename):
    device = 'cpu'
    model = attempt_load(model_filename, map_location=device) # load FP32 model
    
    model_children = list(model.children())

    if output_filename is not None:
        image = cv2.imread(input_filename)  # BGR
        image = cv2.resize(image, (input_size, input_size))
        image = image[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(device)
        image = image.float()
        image /= 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        
        save_output = SaveOutput()

        for l in model_children[0]:
            if isinstance(l, Conv):
                l.act.inplace=False
            elif isinstance (l, Concat):
                pass
            elif isinstance (l, MP):
                pass
            elif isinstance (l, SPPCSPC):
                l.cv2.act.inplace = False
                l.cv1.act.inplace = False
                l.cv3.act.inplace = False
                l.cv4.act.inplace = False
                l.cv5.act.inplace = False
                l.cv6.act.inplace = False
                l.cv7.act.inplace = False
            elif isinstance(l, RepConv):
                if hasattr(l, 'rbr_reparam') == False:
                    l.fuse_repvgg_block()
                l.act.inplace = False
            elif isinstance (l, torch.nn.Upsample):
                pass
            elif isinstance(l, Detect):
                pass
            elif isinstance(l, IDetect):
                pass
            elif isinstance(l, IKeypoint):
                for i in range(l.nl):
                    for k in l.m_kpt[i]:
                        if isinstance(k, Conv):
                            k.act.inplace=False
            else:
                print('layer type not found')
        
        for l in model_children[0]:
            if isinstance(l, Conv):
                l.conv.register_forward_hook(save_output)
                l.act.register_forward_hook(save_output)
            elif isinstance (l, Concat):
                l.register_forward_hook(save_output)
            elif isinstance (l, MP):
                l.m.register_forward_hook(save_output)
            elif isinstance (l, SPPCSPC):
                l.cv2.conv.register_forward_hook(save_output)
                l.cv2.act.register_forward_hook(save_output)
                l.cv1.conv.register_forward_hook(save_output)
                l.cv1.act.register_forward_hook(save_output)
                l.cv3.conv.register_forward_hook(save_output)
                l.cv3.act.register_forward_hook(save_output)
                l.cv4.conv.register_forward_hook(save_output)
                l.cv4.act.register_forward_hook(save_output)
                l.cv5.conv.register_forward_hook(save_output)
                l.cv5.act.register_forward_hook(save_output)
                l.cv6.conv.register_forward_hook(save_output)
                l.cv6.act.register_forward_hook(save_output)
                l.cv7.conv.register_forward_hook(save_output)
                l.cv7.act.register_forward_hook(save_output)
            elif isinstance(l, RepConv):
                l.rbr_reparam.register_forward_hook(save_output)
                l.act.register_forward_hook(save_output)
            elif isinstance (l, torch.nn.Upsample):
                l.register_forward_hook(save_output)
            elif isinstance(l, Detect):
                print('Detect')
            elif isinstance(l, IDetect):
                print('IDetect')
            elif isinstance(l, IKeypoint):
                for i in range(l.nl):
                    l.m[i].register_forward_hook(save_output)
                    for k in l.m_kpt[i]:
                        if isinstance(k, Conv):
                            k.conv.register_forward_hook(save_output)
                            k.act.register_forward_hook(save_output)
                        elif isinstance(k, torch.nn.Conv2d):
                            k.register_forward_hook(save_output)
            else:
                print('layer type not found')

        pred = model(image)[0]
        
        if output_filename is not None:
            with open(output_filename, 'wb') as f:
                np.array(len(save_output.outputs)).tofile(f)
                for o in save_output.outputs:
                    f.write(o[0].encode())
                    np.array(o[1], dtype=np.int32).tofile(f)
                    np.array(o[2]).tofile(f)
                    np.array(o[3], dtype=np.int32).tofile(f)
                    np.array(o[4]).tofile(f)
                    if len(o) == 9:
                        np.array(o[5], dtype=np.int32).tofile(f)
                        np.array(o[6]).tofile(f)
                        np.array(o[7], dtype=np.int32).tofile(f)
                        np.array(o[8]).tofile(f)


def show_usage():
    print('<debug.py> '
          '-i/--input_filename input filename | '
          '-m/--model_filename model filename | '
          '-s/--input_size input size | '
          '-o/--output_filename output filename | '
          '-h/--help')


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:m:s:o:', ['help',
                                                               'input_filename=',
                                                               'model_filename=',
                                                               'input_size=',
                                                                'output_filename='])
    except getopt.GetoptError as err:
        print(str(err))
        show_usage()
        sys.exit(2)

    input_filename = None
    model_filename = None
    input_size = 640
    output_filename = None

    for o, a in opts:
        if o in ('-h', '--help'):
            show_usage()
            sys.exit()
        elif o in ('-i', '--input_filename'):
            input_filename = a
        elif o in ('-m', '--model_filename'):
            model_filename = a
        elif o in ('-s', '--input_size'):
            input_size = int(a)
        elif o in ('-o', '--output_filename'):
            output_filename = a

    main(input_filename, model_filename, input_size, output_filename)
