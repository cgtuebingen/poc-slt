# this is an approach that we use a heuristic to generate our masking
# every case of this heuristic is chosen randomly in each training epoch
# it is a mix of random masking given a masking ratio and custom cut

from einops import rearrange
import torch
from src.utils import generate_random_mask as gr_mask


def mask_heuristic(
    masking_choice: int,
    masking_ratio: torch.float32,
    batch_size,
    number_of_sub_voxels,
    target_resolution,
    given_device,
):
    assert masking_choice <= 34
    random_masking_max = 11
    if (masking_choice <= random_masking_max) and (0 <= masking_choice):
        # do random_masking
        mask_all_bool, num_mask_all = gr_mask.generate_random_mask_for_all(
            number_of_sub_voxels, batch_size, masking_ratio=masking_ratio
        )

    elif masking_choice > random_masking_max:
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
        # set to zero
        all_bool = torch.zeros(
            [
                batch_size,
                number_of_sub_voxels_x,
                number_of_sub_voxels_y,
                number_of_sub_voxels_z,
            ],
            dtype=torch.bool,
            device=given_device,
        )

        if masking_choice == (random_masking_max + 1):
            all_bool[:, :, 0, :] = True
            # num_true = torch.count_nonzero(all_bool)
            # print("\n num_true: ", num_true)

        elif masking_choice == (random_masking_max + 2):
            all_bool[:, :, 1, :] = True

        elif masking_choice == (random_masking_max + 3):
            all_bool[:, :, 2, :] = True

        elif masking_choice == (random_masking_max + 4):
            all_bool[:, :, 3, :] = True

        elif masking_choice == (random_masking_max + 5):
            all_bool[:, :, 0:2, :] = True  # top-half

        elif masking_choice == (random_masking_max + 6):
            all_bool[:, :, 2:4, :] = True  # bottom-half

        elif masking_choice == (random_masking_max + 7):
            all_bool[:, 0:2, 0:2, 0:2] = True  # "front-bottom-right"
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 8):
            all_bool[:, 0:2, 0:2, 2:4] = True  # "back-bottom-right"
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 9):
            all_bool[:, 0:2, 2:4, 0:2] = True  # "front-top-right"
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 10):
            all_bool[:, 0:2, 2:4, 2:4] = True  # "back-top-right"
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 11):
            all_bool[:, 2:4, 0:2, 0:2] = True  # "front-bottom-left"
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 12):
            all_bool[:, 2:4, 0:2, 2:4] = True  # "back-bottom-left"
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 13):
            all_bool[:, 2:4, 2:4, 0:2] = True  # "front-top-left"
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 14):
            all_bool[:, 2:4, 2:4, 2:4] = True  # "back-top-left"
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 15):
            all_bool[:, 0:2, :, :] = True  # right half
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 16):
            all_bool[:, 0, :, :] = True  # right half
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 17):
            all_bool[:, 1, :, :] = True  # right half
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 18):
            all_bool[:, 2, :, :] = True  # right half
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 19):
            all_bool[:, 3, :, :] = True  # right half
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 20):
            all_bool[:, :, :, 0:2] = True  # left half
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 21):
            all_bool[:, :, :, 0] = True  # left half
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 22):
            all_bool[:, :, :, 1] = True  # left half
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 23):
            all_bool[:, :, :, 2] = True  # left half
            all_bool = torch.logical_not(all_bool)

        elif masking_choice == (random_masking_max + 23):
            all_bool[:, :, :, 3] = True  # left half
            all_bool = torch.logical_not(all_bool)

        else:
            print("\n masking_choice: ", masking_choice)
            raise RuntimeError("masking choice is invalid!")

        mask_all_bool = rearrange(all_bool, "B D H W -> B (D H W)")

    else:
        print("\n masking_choice: ", masking_choice)
        raise RuntimeError("masking choice is invalid!")
    return mask_all_bool
