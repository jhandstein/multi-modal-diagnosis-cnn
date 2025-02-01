from src.data_management.mri_image_files import MriImageFile
from src.utils.config import FeatureMapType

# https://www.baeldung.com/cs/convolutional-layer-size
class ConvCalculator:
    def __init__(self, kernel_size=5, stride=1, padding=1, pool_kernel_size=2, pool_stride=2):
        """Initialize calculator with network parameters."""
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
    
    def calculate_network_output_size(self, input_size):
        """Calculate final output size after 4 convolutions and 2 maxpools."""
        # Conv1
        size = self._compute_conv_size(input_size)
        # Maxpool1
        size = self._compute_maxpool_size(size)
        # Conv2
        size = self._compute_conv_size(size)
        # Maxpool2
        size = self._compute_maxpool_size(size)
        # Conv3
        size = self._compute_conv_size(size)
        # Conv4
        size = self._compute_conv_size(size)
        return size
    
    def test_file(self, subject_id: int, feature_map: FeatureMapType) -> None:
        """Test the ConvCalculator with a specific file."""
        image_file = MriImageFile(subject_id, feature_map)
        input_size = image_file.get_size()[1:]
        output_size = self.calculate_network_output_size(input_size)
        print(f"Testing file: {image_file.file_path}")
        print(f"File parameters: {feature_map.modality_label}, {feature_map.label}")
        print(f"Input size feature map: {input_size}")
        print(f"Output size feature map (for FC layer): {output_size}")

    def _compute_conv_size(self, input_size):
        """Calculate output size after convolution for arbitrary dimensions."""
        return tuple((dim - self.kernel_size + 2 * self.padding) // self.stride + 1 
                        for dim in input_size)

    def _compute_maxpool_size(self, input_size): 
        """Calculate output size after maxpooling for arbitrary dimensions.""" 
        # remove batch dimension 
        return tuple((dim - self.pool_kernel_size) // self.pool_stride + 1 for dim in input_size)