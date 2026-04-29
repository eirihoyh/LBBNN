from __future__ import annotations


def compute_output_dimensions(
    n_layers: int,
    input_width: int,
    input_height: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> tuple[int, int]:
    """Compute the spatial output size of a stack of same-parameter conv layers.

    Args:
        n_layers: Number of convolutional layers.
        input_width: Width of the input feature map.
        input_height: Height of the input feature map.
        kernel_size: Kernel size (assumed square).
        stride: Convolution stride.
        padding: Zero-padding added on both sides.

    Returns:
        A ``(width, height)`` tuple for the final output feature map.
    """
    width, height = input_width, input_height
    for _ in range(n_layers):
        width = (width - kernel_size + 2 * padding) // stride + 1
        height = (height - kernel_size + 2 * padding) // stride + 1
    return width, height
