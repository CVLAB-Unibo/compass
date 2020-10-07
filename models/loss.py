import torch
import torch.nn as nn


class ThetaBorisovLoss(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.device = device
        self.eps = 1e-7

    def forward(self, tensor_mat_a, tensor_mat_b):

        """
         Compute difference between rotation matrix as in http://boris-belousov.net/2016/12/01/quat-dist/ on batch tensor
         :param tensor_mat_a: a tensor of rotation matrices in format [B x 3 X 3]
         :param tensor_mat_b: a tensor of rotation matrices in format [B x 3 X 3]
         :return: B values in range [0, 3.14]
         """

        mat_rotation = torch.bmm(tensor_mat_a, tensor_mat_b.transpose(2, 1))
        identity = torch.eye(3, requires_grad=True, device=self.device)

        identity = identity.reshape((1, 3, 3))
        batch_identity = identity.repeat(tensor_mat_a.size(0), 1, 1)

        trace = ((batch_identity * mat_rotation).sum(dim=(1, 2)) - 1) * 0.5

        trace = trace.clamp(min=-1 + self.eps, max=1 - self.eps)
        angles = torch.acos(trace)

        return angles


class ChamferLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, points_src, points_trg):
        """
        Compute the Chamfer distances between two points set
        :param points_src: source input points [B X NUM_POINTS_ X CHANNELS]
        :param points_trg: target input points [B X NUM_POINTS_ X CHANNELS]
        :return two tensors, one for each set, containing the minimum squared euclidean distance between a point
        and its closest point in the other set
        """

        x, y = points_src, points_trg
        bs, num_points, points_dim = x.size()

        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))

        diag_indices = torch.arange(0, num_points).type(torch.cuda.LongTensor) if points_src.device.type == 'cuda' else torch.arange(0, num_points).type(torch.LongTensor)

        x_squared = xx[:, diag_indices, diag_indices].unsqueeze(1).expand_as(xx)
        y_squared = yy[:, diag_indices, diag_indices].unsqueeze(1).expand_as(yy)

        distances = (x_squared.transpose(2, 1) + y_squared - 2 * zz)

        return distances.min(1)[0], distances.min(2)[0]
        