import json
import os

import pandas as pd

from config.ConfigMixin import ConfigMixin
from config.JSONSaver import JSONsaver


class DataSplit(ConfigMixin):
    """Local CSV/TSV split used by the slot training pipeline."""

    def __init__(
        self,
        path: str,
        id_col: str,
        attr_cols: list,
        label_cols: list,
        skip_labels: dict = None,
        ignore_ids: list = None,
        verbose: bool = True,
    ):
        self.path = path
        self.id_col = id_col
        self.attr_cols = attr_cols
        self.label_cols = label_cols
        self.skip_labels = skip_labels
        self.ignore_ids = ignore_ids

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"{self.path} does not exist.")

        df = pd.read_csv(
            self.path,
            sep="\t" if self.path.endswith(".tsv") else ",",
            dtype={"case_id": str, "slide_id": str, "id": str},
        )
        self.data = self._to_samples(df)
        self.num_folds = len(self.data[0]["folds"]) if self.data and "folds" in self.data[0] else 0

        if verbose:
            print(f"Loaded split from {self.path} with {len(self.data)} samples and {self.num_folds} folds assigned.")

    def _to_samples(self, df):
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        df[self.label_cols] = df[self.label_cols].fillna("")

        if self.skip_labels:
            for label_name, skip_vals in self.skip_labels.items():
                df = df[~df[label_name].isin(skip_vals)]

        samples = []
        for sample_id in df[self.id_col].unique():
            sample_df = df[df[self.id_col] == sample_id]
            sample = {"id": str(sample_id)}

            if self.ignore_ids and str(sample_id) in [str(x) for x in self.ignore_ids]:
                continue

            unique_label_sets = sample_df.drop_duplicates(subset=self.label_cols, keep="first")
            if len(unique_label_sets) != 1:
                raise ValueError(f"Inconsistent labels found for sample {sample_id}:\n{unique_label_sets}")
            sample["labels"] = {col: sample_df[col].tolist()[0] for col in self.label_cols}

            fold_columns = [col for col in sample_df.columns if col.startswith("fold_")]
            fold_columns.sort(key=lambda x: int(x.split("_")[-1]))
            sample["folds"] = []
            for col in fold_columns:
                folds_for_sample = set(sample_df[col].values)
                if len(folds_for_sample) != 1:
                    raise ValueError(f"Inconsistent {col} found for sample ID {sample_id}.")
                sample["folds"].append(folds_for_sample.pop())

            for attr_col in self.attr_cols:
                sample[attr_col] = sample_df[attr_col].tolist()

            samples.append(sample)
        return samples

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        preview = "\n".join(str(sample) for sample in self.data[:5])
        return f"Split with {len(self.data)} samples and {self.num_folds} folds assigned.\nFirst 5 samples:\n{preview}"

    def save(self, export_to, row_divisor=None):
        os.makedirs(os.path.dirname(export_to), exist_ok=True)
        if export_to.endswith(".json"):
            with open(export_to, "w") as f:
                json.dump(self.data, f, indent=4, cls=JSONsaver)
            return

        rows = []
        for sample in self.data:
            row = {k: v for k, v in sample.items() if k not in ["id", "labels", "folds"]}
            row.update(sample["labels"])
            for fold, fold_assignment in enumerate(sample["folds"]):
                row[f"fold_{fold}"] = fold_assignment

            for key, value in list(row.items()):
                if isinstance(value, list) and len(set(value)) == 1:
                    row[key] = value[0]

            if row_divisor is None:
                rows.append(row)
            else:
                for unique_val in sample[row_divisor]:
                    new_row = row.copy()
                    new_row[row_divisor] = unique_val
                    rows.append(new_row)

        df = pd.DataFrame(rows)
        if export_to.endswith(".csv"):
            df.to_csv(export_to, index=False)
        elif export_to.endswith(".tsv"):
            df.to_csv(export_to, sep="\t", index=False)
        else:
            raise ValueError(f"Export path must end in .json, .csv, or .tsv. Received {export_to}.")

    def replace_folds(self, replace_from, replace_to, selected_ids=None, selected_folds=None):
        reassigned = {fold_idx: 0 for fold_idx in range(self.num_folds)}
        selected_ids = set(selected_ids) if selected_ids is not None else None
        selected_folds = set(selected_folds) if selected_folds is not None else None

        for sample in self.data:
            if selected_ids is not None and sample["id"] not in selected_ids:
                continue
            for fold_idx, current_assignment in enumerate(sample["folds"]):
                if selected_folds is not None and fold_idx not in selected_folds:
                    continue
                if current_assignment == replace_from:
                    sample["folds"][fold_idx] = replace_to
                    reassigned[fold_idx] += 1

        for fold_idx, count in reassigned.items():
            if count > 0:
                print(f"Reassigned {count} samples in fold {fold_idx} from {replace_from} to {replace_to}.")
