import torch
def concatenate_for_given_dim(tensor_1: torch.Tensor, tensor_2: torch.Tensor, cat_dim: int) -> torch.Tensor:
    # assert tensor_1.shape == tensor_2.shape
    concatenated_result = torch.cat((tensor_1, tensor_2), dim=cat_dim).to(device=tensor_1.device)

    return concatenated_result