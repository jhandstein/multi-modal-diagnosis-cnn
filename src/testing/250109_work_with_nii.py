from data_management.mri_image_files import MriImageFile
from src.utils.config import NAKO_PATH

subject_id = 100000
scan_type = "anat"
prob_type = "GM"

file = MriImageFile(subject_id, scan_type, prob_type)


file.print_stats()

img = file.load_middle_slice()