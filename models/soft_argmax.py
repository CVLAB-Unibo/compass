from __future__ import division

import torch

def _make_radial_cube(width, height, depth, cx, cy, cz, fn, cube_side=10.0):
    """
    Returns a cube, where grid[i,j,k] = fn((i**2 + j**2 + k**2)**0.5)

    :param width: Width of the cube to return
    :param height: Height of the cube to return
    :param cx: x center
    :param cy: y center
    :param cz: z center
    :param fn: The function to apply
    :return:
    """
    # The length of cx and cy is the number of channels we need
    channels = cx.size(0)

    # Make the shape [channels, depth, height, width]
    cx = cx.repeat(height, width, depth, 1).permute(3, 2, 0, 1)
    cy = cy.repeat(height, width, depth, 1).permute(3, 2, 0, 1)
    cz = cz.repeat(height, width, depth, 1).permute(3, 2, 0, 1)

    # Aren't the following lines wrong? In ys it should be (1, height, 1) and so on, right?
    if (cx.device.type == 'cuda'):
        xs = torch.arange(width).view((1, 1, width)).repeat(channels, depth, height, 1).float().cuda()
        ys = torch.arange(height).view((1, width, 1)).repeat(channels, depth, 1, width).float().cuda()
        zs = torch.arange(width).view((depth, 1, 1)).repeat(channels, 1, height, width).float().cuda()
    else:
        xs = torch.arange(width).view((1, 1, width)).repeat(channels, depth, height, 1).float()
        ys = torch.arange(height).view((1, width, 1)).repeat(channels, depth, 1, width).float()
        zs = torch.arange(width).view((depth, 1, 1)).repeat(channels, 1, height, width).float()

    delta_xs = xs - cx
    delta_ys = ys - cy
    delta_zs = zs - cz

    dists = torch.sqrt((delta_ys ** 2) + (delta_xs ** 2) + (delta_zs ** 2))
    #print(dists)

    # apply the function to the cube and return it
    return fn(dists, cube_side)


def _parzen_scalar(delta, width):
    """For reference"""
    del_ovr_wid = math.abs(delta) / width
    if delta <= width/2.0:
        return 1 - 6 * (del_ovr_wid ** 2) * (1 - del_ovr_wid)
    elif delta <= width:
        return 2 * (1 - del_ovr_wid) ** 3


def _parzen_torch(dists, width):
    """
    A PyTorch version of the parzen window that works a grid of distances about some center point.
    See _parzen_scalar to see the 

    :param dists: The grid of distances
    :param window: The width of the parzen window
    :return: A 2d grid, who's values are a (radial) parzen window
    """

    hwidth = width / 2.0
    del_ovr_width = dists / hwidth

    near_mode = (dists <= hwidth/2.0).float()
    in_tail = ((dists > hwidth/2.0) * (dists <= hwidth)).float()

    return near_mode * (1 - 6 * (del_ovr_width ** 2) * (1 - del_ovr_width)) \
        + in_tail * (2 * ((1 - del_ovr_width) ** 3))


def _uniform_window(dists, width):
    """
    A (radial) uniform window function
    :param dists: A grid of distances
    :param width: A width for the window
    :return: A 2d grid, who's values are 0 or 1 depending on if it's in the window or not
    """
    hwidth = width / 2.0
    return (dists <= hwidth).float()


def _identity_window(dists, width):
    """
    An "identity window". (I.e. a "window" which when multiplied by, will not change the input).
    """
    return torch.ones(dists.size())


class SoftArgmax3D(torch.nn.Module):
    """
    """
    def __init__(self, base_index=0, step_size=1, cube_fn=None, cube_width=10, softmax_temp=1.0):
        """
        """
        super(SoftArgmax3D, self).__init__()

        self.base_index = base_index
        self.step_size = step_size

        self.softmax = torch.nn.Softmax(dim=2)
        self.softmax_temp = softmax_temp

        self.cube_type = cube_fn
        self.cube_width = cube_width
        self.cube_fn = _identity_window

        if cube_fn == "Parzen":
            self.cube_fn = _parzen_torch
        elif cube_fn == "Uniform":
            self.cube_fn = _uniform_window

    def _softmax_2d(self, x, temp):
        """
        For the lack of a true 2D softmax in pytorch, we reshape each image from (C, W, H) to (C, W*H) and then
        apply softmax, and then restore the original shape.

        :param x: A 5D tensor of shape (B, C, W, H, D) to apply softmax across the W and H dimensions
        :param temp: A scalar temperature to apply as part of the softmax function
        :return: Softmax(x, dims=(2,3))
        """
        B, C, W, H, D = x.size()
        x_flat = x.view((B, C, W*H*D)) / temp
        x_softmax = self.softmax(x_flat)

        return x_softmax.view((B, C, W, H, D))

    def forward(self, x):
        #TODO Why 2d?
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:

        SoftArgMax2d(x) = (\sum_i \sum_j (i * softmax2d(x)_ij), \sum_i \sum_j (j * softmax2d(x)_ij))

        :param x: The input to the soft arg-max layer
        :return: Output of the 2D soft arg-max layer, x_coords and y_coords, in the shape (B, C, 2), which are the soft
            argmaxes per channel
        """
        # Compute windowed softmax
        # Compute windows using a batch_size of "batch_size * channels"
        batch_size, channels, depth, height, width = x.size()

        argmax = torch.argmax(x.view(batch_size * channels, -1), dim=1)

        argmax_x = torch.floor(torch.div(argmax.float(), torch.mul(height, float(depth)).float()))
        argmax_y = torch.remainder(torch.floor(torch.div(argmax.float(), float(depth))), height)
        argmax_z = torch.remainder(argmax.float(), float(depth)).float()
        #print(x)
        windows = _make_radial_cube(width, height, depth, argmax_z, argmax_y, argmax_x, self.cube_fn, self.cube_width)
        windows = windows.view(batch_size, channels, height, width, depth).cuda() if x.device.type == 'cuda' else windows.view(batch_size, channels, height, width, depth)

        smax = self._softmax_2d(x, self.softmax_temp) * windows
        smax = smax / torch.sum(smax.view(batch_size, channels, -1), dim=2).view(batch_size, channels, 1, 1, 1)

        # compute x index (sum over y and z axes, produce with indices and then sum over x axis for the expectation)
        x_end_index = self.base_index + width * self.step_size
        x_indices = torch.arange(start=self.base_index, end=x_end_index, step=self.step_size).cuda() if x.device.type == 'cuda' else torch.arange(start=self.base_index, end=x_end_index, step=self.step_size)
        x_coords = torch.sum(torch.sum(smax, dim=(3, 4)) * x_indices, 2)

        # compute y index (sum over x and z axes, produce with indices and then sum over y axis for the expectation)
        y_end_index = self.base_index + height * self.step_size
        y_indices = torch.arange(start=self.base_index, end=y_end_index, step=self.step_size).cuda() if x.device.type == 'cuda' else torch.arange(start=self.base_index, end=y_end_index, step=self.step_size)
        y_coords = torch.sum(torch.sum(smax, dim=(2, 4)) * y_indices, 2)

        # compute z index (sum over x axis, produce with indices and then sum over y axis for the expectation)
        z_end_index = self.base_index + depth * self.step_size
        z_indices = torch.arange(start=self.base_index, end=z_end_index, step=self.step_size).cuda() if x.device.type == 'cuda' else torch.arange(start=self.base_index, end=z_end_index, step=self.step_size)
        z_coords = torch.sum(torch.sum(smax, dim=(2, 3)) * z_indices, 2)

        # Put the x coords, y coords and z coords (shape (B,C)) into an output with shape (B,C,3)
        return torch.cat([torch.unsqueeze(x_coords, 2),
                          torch.unsqueeze(y_coords, 2),
                          torch.unsqueeze(z_coords, 2)], dim=2)
