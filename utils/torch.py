import sys
import torch


def get_gpu_device(id):
    if torch.cuda.is_available():
        device_gpu = torch.device("cuda:" + str(id))
    else:
        raise RuntimeError('No gpu device found.')

    return device_gpu


def get_cpu_device():
    device_cpu = torch.device("cpu")

    return device_cpu


def print_network_module(network, stream=sys.stdout):

    print(format(network), file=stream, flush=False)

    for name, param in network.named_parameters():
        if 'kernel' in name:
            layer_str = "Name: {} - Shape {}".format(name, param.transpose(2, 1).shape)
            print(layer_str, file=stream, flush=False)


def b_get_rotation_matrices_from_euler_angles_on_tensor(alphas, betas, gammas, device):

    zs = torch.zeros(alphas.shape, device=device, requires_grad=True)
    os = torch.ones(alphas.shape, device=device, requires_grad=True)

    def z(a):
        first_row = torch.stack([torch.cos(a), torch.sin(a), zs], dim=1)
        second_row = torch.stack([-torch.sin(a), torch.cos(a), zs], dim=1)
        third_row = torch.stack([zs, zs, os], dim=1)
        mat_z = torch.stack([first_row, second_row, third_row], dim=1)

        return mat_z

    def y(a):
        f_rw = torch.stack([torch.cos(a), zs, -torch.sin(a)], dim=1)
        s_rw = torch.stack([zs, os, zs], dim=1)
        t_rw = torch.stack([torch.sin(a), zs, torch.cos(a)], dim=1)

        mat_y = torch.stack([f_rw, s_rw, t_rw], dim=1)

        return mat_y

    return torch.bmm(torch.bmm(z(gammas), y(betas)), z(alphas))


def load_models_from_ckp(path_checkpoint, model):
    """
    Function to load values in checkpoint file.
    :param path_checkpoint: path to ckp file
    :param model: model for which to load the weights
    :return:
    """

    if path_checkpoint is not None:
        dict_ckp = torch.load(path_checkpoint, map_location=torch.device('cpu'))

        print("Loaded model from: {}".format(path_checkpoint))

        for key in dict_ckp:
            print("{}".format(key))

        model.load_state_dict(dict_ckp)
        return True

    return False


def rotate_batch_cloud(b_points, b_mats):

    """
    :param points: a tensor of point in row vector format [B X N X 3]
    :param mats: a tensor of 3 x 3 rotation matrices format [B X 3 X 3]
    :return:
    """

    # Tranpose rotation matrices as we multiply row vector points
    return torch.bmm(b_points, b_mats.transpose(2, 1).contiguous())