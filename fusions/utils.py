import torch


def concat_outputs(
    first_feature: torch.Tensor,
    second_feature: torch.Tensor,
    first_feature_lengths: torch.Tensor,
    second_feature_lengths: torch.Tensor,
    device="cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    (batch_size, _, feature_size) = first_feature.shape
    new_lengths = first_feature_lengths + second_feature_lengths
    joint_outputs = first_feature.new_zeros(
        (batch_size, new_lengths.max().item(), feature_size)
    )

    max_first_length = first_feature.shape[1]
    first_batch_ids = torch.arange(batch_size, device=device).unsqueeze(1)
    first_batch_ids = first_batch_ids.expand(batch_size, max_first_length)

    first_position_ids = torch.arange(max_first_length, device=device).unsqueeze(0)
    first_position_ids = first_position_ids.expand(batch_size, max_first_length)

    first_valid_positions = first_position_ids < first_feature_lengths.unsqueeze(1)

    max_second_length = second_feature.shape[1]

    second_batch_ids = torch.arange(batch_size, device=device).unsqueeze(1)
    second_batch_ids = second_batch_ids.expand(batch_size, max_second_length)

    second_position_ids = torch.arange(max_second_length, device=device).unsqueeze(0)
    second_position_ids = second_position_ids.expand(batch_size, max_second_length)

    second_valid_positions = second_position_ids < second_feature_lengths.unsqueeze(1)

    second_joint_positions = first_feature_lengths.unsqueeze(1) + second_position_ids

    joint_outputs[
        first_batch_ids[first_valid_positions],
        first_position_ids[first_valid_positions],
    ] = first_feature[first_valid_positions]

    joint_outputs[
        second_batch_ids[second_valid_positions],
        second_joint_positions[second_valid_positions],
    ] = second_feature[second_valid_positions]

    return (joint_outputs, new_lengths)


if __name__ == "__main__":
    from torch.nn.utils.rnn import pad_sequence

    first = [
        torch.tensor([1, 1, 1, 1, 1]),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]),
        torch.tensor([1, 1, 1]),
    ]

    second = [
        torch.tensor([2, 2, 2]),
        torch.tensor([2, 2]),
        torch.tensor([2, 2, 2, 2, 2]),
    ]

    padded_first = pad_sequence(first, batch_first=True).unsqueeze(-1)
    padded_second = pad_sequence(second, batch_first=True).unsqueeze(-1)

    (output, output_lengths) = concat_outputs(
        first_feature=padded_first,
        second_feature=padded_second,
        first_feature_lengths=torch.tensor([5, 8, 3]),
        second_feature_lengths=torch.tensor([3, 2, 5]),
    )

    print(output)
    print(output_lengths)
