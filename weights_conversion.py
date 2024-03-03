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


def convert_conv(l):
    weights = []
    # write convolutional layer bias
    if l.conv.bias is not None:
        weights.extend(l.conv.bias.detach().numpy())
    else:
        weights.extend(np.zeros(l.conv.out_channels))
    # write batch normalization layer weights
    if hasattr(l, 'bn'):
        weights.extend(l.bn.weight.detach().numpy())
        # write batch normalization layer rolling mean
        weights.extend(l.bn.running_mean.detach().numpy())
        # write batch normalization layer rolling variance
        weights.extend(l.bn.running_var.detach().numpy())
    # write convolutional layer weights
    weights.extend(l.conv.weight.detach().numpy().flatten())

    return weights


def convert_sppcspc(l):
    weights = []
    weights.append(convert_conv(l.cv2))
    weights.append(convert_conv(l.cv1))
    weights.append(convert_conv(l.cv3))
    weights.append(convert_conv(l.cv4))
    weights.append(convert_conv(l.cv5))
    weights.append(convert_conv(l.cv6))
    weights.append(convert_conv(l.cv7))

    return weights


def convert_repconv(l):
    if hasattr(l, 'rbr_reparam') == False:
        l.fuse_repvgg_block()
    weights = []
    if l.rbr_reparam:
        if l.rbr_reparam.bias is not None:
            weights.extend(l.rbr_reparam.bias.detach().numpy())
        else:
            weights.extend(np.zeros(l.rbr_reparam.out_channels))
        # write convolutional layer weights
        weights.extend(l.rbr_reparam.weight.detach().numpy().flatten())
    else:
        pass

    return weights


def convert_yolo(l):
    weights = []
    for i in range(l.nl):
        w = []
        if l.m[i].bias is not None:
            w.extend(l.m[i].bias.detach().numpy())
        else:
            w.extend(np.zeros(l.m[i].out_channels))
        # write convolutional layer weights
        # permuted_weights = l.m[i].weight.permute(0, 2, 3, 1)
        # w.extend(permuted_weights.detach().numpy().flatten())
        w.extend(l.m[i].weight.detach().numpy().flatten())

        weights.append(w)

    return weights


def read_layer_indices(s, layer_count):
    idx = []
    if s:
        for v in s.split(','):
            if '-' in v:
                aux = v.split('-')
                first_index = 0
                last_index = layer_count
                if len(aux[0]) > 0:
                    first_index = int(aux[0])
                if len(aux[1]) > 0:
                    last_index = int(aux[1]) + 1
                idx.extend(range(first_index, last_index))
            else:
                idx.append(int(v))
    else:
        return idx

    idx = list(dict.fromkeys(idx))
    idx.sort()
    return idx


def main(input_filename, layers, append_filename, output_filename, save_weights):
    MAJOR_VERSION = 0
    MINOR_VERSION = 2
    PATCH_VERSION = 5
    SEEN = 1

    device = select_device('cpu', batch_size=1)

    model = Ensemble()
    attempt_download(input_filename)
    ckpt = torch.load(input_filename, map_location=device)  # load
    model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        model = model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % input_filename)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))

    indices = read_layer_indices(layers, len(model.model))

    prefix = [MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION, SEEN]
    weights = []
    repconv_weights = []
    yolo_weights = []
    counter = 0
    for n, l in enumerate(model.model):
        if len(indices) == 0 or n in indices:
            print('saving layer #' + str(n) + ' weights...')
            print(l)
            if isinstance(l, Conv):
                layer_weights = convert_conv(l)
                if save_weights:
                    layer_weights_filename = path.join(path.dirname(output_filename), "layer_" + str(counter) + ".txt")
                    counter += 1
                    with open(layer_weights_filename, "w") as txt_file:
                        for v in layer_weights:
                            txt_file.write(str(v) + "\n")
                weights.extend(layer_weights)
            elif isinstance(l, SPPCSPC):
                sppcspc_weights = convert_sppcspc(l)
                if save_weights:
                    for sppcspc_w in sppcspc_weights:
                        layer_weights_filename = path.join(path.dirname(output_filename), "layer_" + str(counter) + ".txt")
                        counter += 1
                        with open(layer_weights_filename, "w") as txt_file:
                            for v in sppcspc_w:
                                txt_file.write(str(v) + "\n")
                for sppcspc_w in sppcspc_weights:
                    weights.extend(sppcspc_w)
            elif isinstance(l, RepConv):
                layer_weights = convert_repconv(l)
                if save_weights:
                    layer_weights_filename = path.join(path.dirname(output_filename), "layer_" + str(counter) + "_r.txt")
                    counter += 1
                    with open(layer_weights_filename, "w") as txt_file:
                        for v in layer_weights:
                            txt_file.write(str(v) + "\n")
                repconv_weights.append(layer_weights)
            elif isinstance(l, Detect):
                yolo_weights = convert_yolo(l)
                if save_weights:
                    for yolo_w in yolo_weights:
                        layer_weights_filename = path.join(path.dirname(output_filename), "layer_" + str(counter) + "_y.txt")
                        counter += 1
                        with open(layer_weights_filename, "w") as txt_file:
                            for v in yolo_w:
                                txt_file.write(str(v) + "\n")
            elif isinstance(l, IDetect):
                yolo_weights = convert_yolo(l)
                if save_weights:
                    for yolo_w in yolo_weights:
                        layer_weights_filename = path.join(path.dirname(output_filename), "layer_" + str(counter) + ".txt")
                        counter += 1
                        with open(layer_weights_filename, "w") as txt_file:
                            for v in yolo_w:
                                txt_file.write(str(v) + "\n")
            else:
                counter += 1
        else:
            print('skipping layer #' + str(n))

    for i in range(len(repconv_weights)):
        weights.extend(repconv_weights[i])
        weights.extend(yolo_weights[i])

    if append_filename:
        copyfile(append_filename, output_filename)
        with open(output_filename, 'ab') as f:
            for w in weights:
                f.write(w.tobytes())
    else:
        with open(output_filename, 'wb') as f:
            for p in prefix[:-1]:
                f.write(p.to_bytes(4, 'little'))
            f.write(prefix[-1].to_bytes(8, 'little'))
            for w in weights:
                f.write(w.tobytes())


def show_usage():
    print('<weights_conversion.py> '
          '-i/--input_filename input filename | '
          '-l/--layers layers\' indices, comma separated, \'-\' indicate range | '
          '-a/--append_filename append weights to this filename | '
          '-o/--output_filename output filename | '
          '-s/--save_weights save layers\' weights | '
          '-h/--help')


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:l:a:o:s', ['help',
                                                                'input_filename=',
                                                                'layers=',
                                                                'append_filename=',
                                                                'output_filename=',
                                                                'save_weights'])
    except getopt.GetoptError as err:
        print(str(err))
        show_usage()
        sys.exit(2)

    input_filename = None
    layers = None
    append_filename = None
    output_filename = None
    save_weights = False

    for o, a in opts:
        if o in ('-h', '--help'):
            show_usage()
            sys.exit()
        elif o in ('-i', '--input_filename'):
            input_filename = a
        elif o in ('-l', '--layers'):
            layers = a
        elif o in ('-a', '--append_filename'):
            append_filename = a
        elif o in ('-o', '--output_filename'):
            output_filename = a
        elif o in ('-s', '--save_weights'):
            save_weights = True

    main(input_filename, layers, append_filename, output_filename, save_weights)
