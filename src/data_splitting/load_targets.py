import pandas as pd

from src.utils.config import NAKO_TABLE_PATH

# TODO: refactor this dict approach
NAKO_TARGETS = {
    "subject_id": "ID",
    "center": "StudZ",
    "age": "basis_age",
    "sex": "basis_sex",
    "phq9_sum": "a_emo_phq9_sum",
    "phq9_cutoff": "a_emo_phq9_cut10", # binary label for PHQ-9 >= 10 -> depression (1) or not (0)
    "gad7_sum": "a_emo_gad7_sum",
    "gad7_cutoff": "a_emo_gad7_cut10", # binary label for GAD-7 >= 10 -> moderate or heavy anxiety symptoms (1) or mild or no anxiety symptoms (0)
    "systolic_blood_pressure": "a_rr_sys",
}


def extract_targets(target: str, subject_ids: list[int]) -> pd.Series:
    # TODO: order of function arguments could be flipped
    """
    Extract the target values for the given subject IDs from the NAKO table.

    Args:
        target: The target to extract ("age", "sex", ...)
        subject_ids: List of subject IDs to extract the target for
    Returns:
        A pandas Series with the subject IDs as index and the target values as values
    """
    nako_table = pd.read_csv(NAKO_TABLE_PATH)
    nako_table.set_index("ID", inplace=True)
    series = nako_table.loc[subject_ids, NAKO_TARGETS[target]]
    if target == "sex":
        # 0 means male, 1 means female
        series = series.map({1: 0, 2: 1})
    return series.sort_index()
