from src.data_management.mri_image_files import MriImageFile
from src.utils.config import FeatureType, ModalityType


def calc_convolved_size(subject_id: int):
    """Print the sizes of the feature maps after 4 convolutions and two maxpools."""
    # Set parameters for the convolutional layers
    kernel_size = 5
    stride = 1
    padding = 1

    # Print the sizes of the feature maps
    fm_map = MriImageFile(subject_id, ModalityType.ANAT, FeatureType.GM)
    input_size = fm_map.get_size()[1:]
    output_size = calculate_2d_conv_ouput_size(input_size, kernel_size, stride, padding)
    print(f"Input size feature map: {input_size}")
    print(f"Output size feature map (for FC layer): {output_size}")
    
    # Print the sizes of the raw images
    fm_raw = MriImageFile(subject_id, ModalityType.RAW, FeatureType.SMRI)
    input_size = fm_raw.get_size()[1:]
    output_size = calculate_2d_conv_ouput_size(input_size, kernel_size, stride, padding)
    print(f"Input size raw feature map: {input_size}")
    print(f"Output size raw feature map (for FC layer): {output_size}")

def calculate_2d_conv_ouput_size(input_size, kernel_size, stride, padding):
    """Calculate the size of the output tensor after 4 convolutions and two maxpools. This function mirrors the forward pass of the ConvBranch2d model."""
    height, width = input_size
    # Convolutional layer 1
    height_1, width_1 = compute_size_after_2d_conv(height, width, kernel_size, stride, padding)
    # Maxpool 1
    height_2, width_2 = compute_size_after_2d_maxpool(height_1, width_1)
    # Convolutional layer 2
    height_3, width_3 = compute_size_after_2d_conv(height_2, width_2, kernel_size, stride, padding)
    # Maxpool 2
    height_4, width_4 = compute_size_after_2d_maxpool(height_3, width_3)
    # Convolutional layer 3
    height_5, width_5 = compute_size_after_2d_conv(height_4, width_4, kernel_size, stride, padding)
    # Convolutional layer 4
    height_6, width_6 = compute_size_after_2d_conv(height_5, width_5, kernel_size, stride, padding)
    return height_6, width_6


# https://www.baeldung.com/cs/convolutional-layer-size
def compute_size_after_2d_conv(height, width, kernel_size, stride, padding):
    """Calculate the size of the output tensor after a 2D convolution"""
    # Convolutional layer 1
    height = (height - kernel_size + 2 * padding) // stride + 1
    width = (width - kernel_size + 2 * padding) // stride + 1
    return height, width

def compute_size_after_2d_maxpool(height, width, kernel_size=2, stride=2):
    """Calculate the size of the output tensor after a 2D maxpool"""
    # Uses the same formula as the convolutional layer with padding=0
    height, width = compute_size_after_2d_conv(height, width, kernel_size, stride, padding=0)
    return height, width
