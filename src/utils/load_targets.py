import pandas as pd

from src.utils.config import NAKO_TABLE_PATH

# refactor this dict approach
NAKO_TARGETS = {
    "subject_id": "ID",
    "center": "StudZ",
    "age": "basis_age",
    "sex": "basis_sex"
}

def extract_target(target: str, subject_ids: list[int]) -> pd.Series:
    nako_table = pd.read_csv(NAKO_TABLE_PATH)
    nako_table.set_index("ID", inplace=True)
    series = nako_table.loc[subject_ids, NAKO_TARGETS[target]]
    if target == "sex":
        series = series.map({1: 0, 2: 1})
    return series.sort_index()
