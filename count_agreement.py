import json
import pandas as pd
import random
from copy import deepcopy

# Reviewers you want in the matrix, including future ones
reviewers = ["leticia", "guilherme", "sergio", "mix", "auto"]

# Files currently available
files = {
    "leticia": "./evaluation_manual/benjamin50_leticia.json",
    "guilherme": "./evaluation_manual/benjamin50_guilherme.json",
    "sergio": "./evaluation_manual/benjamin50_sergio.json",
    "mix": "./evaluation_manual/benjamin50_mix.json",
    "auto": "./evaluation_manual/benjamin50_auto.json",
}

data = {}
for reviewer, path in files.items():
    with open(path, "r", encoding="utf-8") as file:
        data[reviewer] = json.load(file)

agreement_matrix = pd.DataFrame(index=reviewers, columns=reviewers, dtype="object")

# Fill the matrix
for row_reviewer in reviewers:
    for col_reviewer in reviewers:
        if row_reviewer not in data or col_reviewer not in data:
            agreement_matrix.loc[row_reviewer, col_reviewer] = None
            continue

        count = 0
        for row_item, col_item in zip(data[row_reviewer], data[col_reviewer]):
            row_eval = row_item.get(row_reviewer, {})
            col_eval = col_item.get(col_reviewer, {})

            true_positive = (
                row_eval.get("answerable") and
                col_eval.get("answerable") and
                row_eval.get("multi-hop") and 
                col_eval.get("multi-hop")
            )

            true_negative = (
                (not row_eval.get("answerable") or not row_eval.get("multi-hop")) and
                (not col_eval.get("answerable") or not col_eval.get("multi-hop"))
            )

            if true_positive or true_negative:
                count += 1

        agreement_matrix.loc[row_reviewer, col_reviewer] = count

print(agreement_matrix)