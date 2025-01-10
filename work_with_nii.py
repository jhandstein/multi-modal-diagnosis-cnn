from data_management.feature_map_files import FeatureMapFile
from utils.config import NAKO_PATH

subject_id = 100000
scan_type = "anat"
prob_type = "GM"

file = FeatureMapFile(subject_id, scan_type, prob_type)


file.print_stats()

img = file.load_middle_slice()