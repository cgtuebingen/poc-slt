def create_log_losses(loss_dict: dict, stage: str) -> dict:
    l1_loss_non_masked = loss_dict["l1_loss_non_masked"]
    l1_loss_masked = loss_dict["l1_loss_masked"]
    l1_loss_empty = loss_dict["l1_loss_empty"]
    # l1_loss_decoded = loss_dict["l1_loss_decoded"]
    l1_loss = loss_dict["l1_loss"]
    log_prefix = stage
    loss_log: dict = {
        f"{log_prefix}/l1_loss_empty": l1_loss_empty.detach(),
        # f"{log_prefix}/l1_loss_decoded": l1_loss_decoded.detach(),
        f"{log_prefix}/l1_loss_non_masked": l1_loss_non_masked.detach(),
        f"{log_prefix}/l1_loss_masked": l1_loss_masked.detach(),
        f"{log_prefix}/l1_loss": l1_loss.detach(),
    }
    return loss_log


def create_log_losses_(loss_dict: dict, stage: str) -> dict:
    l1_loss_non_empty = loss_dict["l1_loss_non_empty"]
    l1_loss_empty = loss_dict["l1_loss_empty"]
    l1_loss = loss_dict["l1_loss"]

    log_prefix = stage
    loss_log: dict = {
        f"{log_prefix}/l1_loss_empty": l1_loss_empty.detach(),
        f"{log_prefix}/l1_loss_non_empty": l1_loss_non_empty.detach(),
        f"{log_prefix}/l1_loss": l1_loss.detach(),

    }
    return loss_log

def create_log_losses_for_given_dict(loss_dict: dict, stage: str):
    log_prefix = stage
    keys = [key for key in loss_dict.keys()]
    loss_log = dict()
    for i in range(len(keys)):
        current_key = keys[i]
        current_value = loss_dict.get(current_key)
        current_key_with_prefix = f"{log_prefix}/" + current_key
        loss_log[current_key_with_prefix] = current_value.detach()
    return loss_log
