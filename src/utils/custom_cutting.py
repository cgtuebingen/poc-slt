import torch
from einops import rearrange

def custom_cut(mode: str, batch_size: int, number_of_sub_voxels: int, given_device) -> torch.bool:
    all_bool = torch.zeros([batch_size, number_of_sub_voxels], dtype=torch.bool, device=given_device)

    quarter_num_cut_sub_voxels = int(number_of_sub_voxels/4)
    quarter_cut_bool = torch.ones([batch_size, quarter_num_cut_sub_voxels], dtype=torch.bool, device=given_device)

    half_num_cut_sub_voxels = int(number_of_sub_voxels/2)
    half_cut_bool = torch.ones([batch_size, half_num_cut_sub_voxels], dtype=torch.bool, device=given_device)

    if mode == 'first-quarter':
        # cut have of the object from top/bottom/left/right
        all_bool[:, :quarter_num_cut_sub_voxels] = quarter_cut_bool

    elif mode == 'second-quarter':
        all_bool[:, quarter_num_cut_sub_voxels: (2*quarter_num_cut_sub_voxels)] = quarter_cut_bool

    elif mode == 'third-quarter':
        all_bool[:, (2 * quarter_num_cut_sub_voxels):(3 * quarter_num_cut_sub_voxels)] = quarter_cut_bool

    elif mode == 'forth-quarter':
        all_bool[:, (3 * quarter_num_cut_sub_voxels):number_of_sub_voxels] = quarter_cut_bool

    elif mode == 'first-half':
        all_bool[:, :half_num_cut_sub_voxels] = half_cut_bool

    elif mode == 'second-half':
        all_bool[:, half_num_cut_sub_voxels: number_of_sub_voxels] = half_cut_bool

    # elif mode == 'top-half':
    #     all_bool[:, half_num_cut_sub_voxels: number_of_sub_voxels] = half_cut_bool

    return all_bool

def custom_mask(mode: str, target_resolution: int, batch_size: int, number_of_sub_voxels: int, given_device: str):
    if target_resolution == 32:
        number_of_sub_voxels_x = 4
        number_of_sub_voxels_y = 4
        number_of_sub_voxels_z = 4

    elif target_resolution == 16:
        number_of_sub_voxels_x = 8
        number_of_sub_voxels_y = 8
        number_of_sub_voxels_z = 8

    elif target_resolution == 8:
        number_of_sub_voxels_x = 16
        number_of_sub_voxels_y = 16
        number_of_sub_voxels_z = 16
    else:
        raise "target resolution is not valid for our setup!"

    all_bool = torch.zeros([batch_size, number_of_sub_voxels_x, number_of_sub_voxels_y, number_of_sub_voxels_z], dtype=torch.bool, device=given_device)

    half_num_cut_sub_voxels = int(number_of_sub_voxels / 2)
    quarter_num_cut_sub_voxels = int(number_of_sub_voxels / 4)
    octant_num_cut_sub_voxels = int(number_of_sub_voxels / 8)

    if mode == 'top-half':
        all_bool[:, :,  0:2, :] = True  # top-half

    elif mode == 'bottom-half':
        all_bool[:, :, 2:4, :] = True  # bottom-half

    elif mode == "front-bottom-right":
        all_bool[:, 0:2, 0:2, 0:2] = True  # 1

        num_true = torch.count_nonzero(all_bool)
        assert (num_true == octant_num_cut_sub_voxels)
        all_bool = torch.logical_not(all_bool)

    elif mode == "back-bottom-right":
        print("\n mode: ", mode)
        all_bool[:, 0:2, 0:2, 2:4] = True  # 2

        num_true = torch.count_nonzero(all_bool)
        print("\n num_true: ", num_true)
        assert (num_true == octant_num_cut_sub_voxels)
        all_bool = torch.logical_not(all_bool)

    elif mode == "front-top-right":
        print("\n mode: ", mode)
        all_bool[:, 0:2, 2:4, 0:2] = True  # 3

        num_true = torch.count_nonzero(all_bool)
        print("\n num_true: ", num_true)
        assert (num_true == octant_num_cut_sub_voxels)
        all_bool = torch.logical_not(all_bool)

    elif mode == "back-top-right":
        print("\n mode: ", mode)
        all_bool[:, 0:2, 2:4, 2:4] = True  # 4

        num_true = torch.count_nonzero(all_bool)
        print("\n num_true: ", num_true)
        assert (num_true == octant_num_cut_sub_voxels)
        all_bool = torch.logical_not(all_bool)

    elif mode == "front-bottom-left":
        print("\n mode: ", mode)
        all_bool[:, 2:4, 0:2, 0:2] = True  # 5

        num_true = torch.count_nonzero(all_bool)
        print("\n num_true: ", num_true)
        assert (num_true == octant_num_cut_sub_voxels)
        all_bool = torch.logical_not(all_bool)

    elif mode == "back-bottom-left":
        print("\n mode: ", mode)
        all_bool[:, 2:4, 0:2, 2:4] = True  # 6

        num_true = torch.count_nonzero(all_bool)
        print("\n num_true: ", num_true)
        assert (num_true == octant_num_cut_sub_voxels)
        all_bool = torch.logical_not(all_bool)

    elif mode == "front-top-left":
        print("\n mode: ", mode)
        all_bool[:, 2:4, 2:4, 0:2] = True  # 7

        num_true = torch.count_nonzero(all_bool)
        print("\n num_true: ", num_true)
        assert (num_true == octant_num_cut_sub_voxels)
        all_bool = torch.logical_not(all_bool)

    elif mode == "back-top-left":
        print("\n mode: ", mode)
        all_bool[:, 2:4, 2:4, 2:4] = True  # 8

        num_true = torch.count_nonzero(all_bool)
        print("\n num_true: ", num_true)
        assert (num_true == octant_num_cut_sub_voxels)
        all_bool = torch.logical_not(all_bool)

    else:
        raise "mode is not valid!"
    all_bool_reshaped = rearrange(all_bool, "B D H W -> B (D H W)")

    return all_bool_reshaped
