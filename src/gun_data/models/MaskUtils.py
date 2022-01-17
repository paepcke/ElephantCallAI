import torch


def create_2d_time_mask(max_time_len: int, time_len: torch.LongTensor, convert_to_bool: bool = False):
    """
    :param max_time_len: max time length of an input sequence of frequency vectors
    :param time_len: a 1D tensor of sequence lengths for each example in the batch. has shape (num_batches,)
    :param convert_to_bool: a flag that allows this function to return a BoolTensor instead of a FloatTensor.
        'True' means an element will be suppressed.
    :return: a 2D mask that excludes computing sequences for
    """
    num_batches = time_len.shape[0]

    target_device = time_len.device

    time_index = torch.arange(max_time_len) + 1
    time_index = torch.unsqueeze(time_index, dim=0).expand(num_batches, max_time_len).to(target_device)
    time_len = torch.unsqueeze(time_len, dim=-1).expand(num_batches, max_time_len)

    time_mask = torch.where(time_index < time_len,
                            torch.ones((1,)).expand(num_batches, max_time_len).to(target_device),
                            torch.zeros((1,)).expand(num_batches, max_time_len).to(target_device))

    if convert_to_bool:
        time_mask = torch.where(time_mask == 0,
                                torch.BoolTensor([True]).bool().expand(num_batches, max_time_len).to(target_device),
                                torch.BoolTensor([False]).bool().expand(num_batches, max_time_len).to(target_device))

    return time_mask


def create_3d_time_mask(num_batches: int, max_time_len: int, time_len: torch.LongTensor):
    time_mask_2d = create_2d_time_mask(num_batches, max_time_len, time_len)

    time_mask_left_2d = torch.reshape(time_mask_2d, (num_batches, max_time_len, 1))
    time_mask_right_2d = torch.reshape(time_mask_2d, (num_batches, 1, max_time_len))

    time_mask_3d = torch.bmm(time_mask_left_2d, time_mask_right_2d)

    return time_mask_3d
