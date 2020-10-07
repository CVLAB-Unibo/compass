import numpy as np
import torch
import torch.nn as nn
from utils import torch as uto

from s2cnn import s2_equatorial_grid, s2_near_identity_grid, S2Convolution, so3_equatorial_grid, SO3Convolution, so3_near_identity_grid


class S2Layer(nn.Module):
    def __init__(self, bandwidths, features, use_equatorial_grid):
        super().__init__()

        self.bandwidths = bandwidths
        self.features = features

        assert len(self.bandwidths) == len(self.features)

        sequence = []

        # S2 layer
        grid_s2 = s2_equatorial_grid(max_beta=0, n_alpha=2 * self.bandwidths[0], n_beta=1) if use_equatorial_grid else s2_near_identity_grid(max_beta=np.pi / 8, n_alpha=8, n_beta=3)
        sequence.append(S2Convolution(self.features[0], self.features[1], self.bandwidths[0], self.bandwidths[1], grid_s2))

        sequence.append(nn.BatchNorm3d(self.features[-1], affine=True))
        sequence.append(nn.ReLU())

        self.sequential = nn.Sequential(*sequence)

    def forward(self, x):  # pylint: disable=W0221
        x = self.sequential(x)

        return x

    def __repr__(self):
        layer_str = ""
        for name, param in self.named_parameters():
            if 'kernel' in name:
                layer_str += "Name: {} - Shape {}".format(name, param.transpose(2, 1).shape) + "\n"

        return super(S2Layer, self).__repr__() + layer_str


if __name__ == "__main__":

    size_bandwidth = 24
    size_channels_input = 4

    num_features = 40

    bandwidths = [size_bandwidth, size_bandwidth, size_bandwidth]
    features = [size_channels_input, num_features, num_features]

    signal_input = torch.randn(1, size_channels_input, 2 * size_bandwidth, 2 * size_bandwidth)

    net = S2Layer(bandwidths,
                features,
                True)

    print(net)

    uto.print_network_module(net)

    net.eval()

    file_object = open("/users/freddy/Desktop/prove_rete/network.csv", "w")

    uto.print_network_module(net, file_object)
    file_object.close()

    import time
    time_start = time.time()
    out = net(signal_input)
    time_elapsed = time.time() - time_start

    print("Time in seconds: {:.3} \n".format(time_elapsed))
