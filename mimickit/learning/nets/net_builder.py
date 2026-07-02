import inspect
import torch

from learning.nets import *

def build_net(net_name, input_dict, activation=torch.nn.ReLU, info=None):
    if (net_name in globals()):
        net_func = globals()[net_name]
        # Forward `info` (e.g. (config, env) for layout-aware nets) only to net
        # builders that accept it; legacy fc nets keep their (input_dict,
        # activation) signature untouched.
        if ("info" in inspect.signature(net_func.build_net).parameters):
            net, out_info = net_func.build_net(input_dict, activation, info=info)
        else:
            net, out_info = net_func.build_net(input_dict, activation)
    else:
        assert(False), "Unsupported net: {}".format(net_name)
    return net, out_info
