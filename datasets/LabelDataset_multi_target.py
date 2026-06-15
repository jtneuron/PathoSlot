import torch

from datasets.BaseDataset import BaseDataset

MULTICLASS = "multiclass"
MULTILABEL_BINARY = "multilabel_binary"
SURVIVAL = "survival"

"""
LabelDataset_multi_target loads one or more labels per sample.
"""
def _safe_int(x, default=-255):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return int(float(x))
    except Exception:
        return default

class LabelDataset_multi_target(BaseDataset):
    def __init__(self, split, task_specs=None, task_names=None, dtype='int', extra_attrs = None):
        super().__init__(split)
        self.task_specs = task_specs or self._legacy_task_specs(task_names)
        self.task_names = task_names
        self.extra_attrs = extra_attrs
        self.dtype = dtype
        
    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        sample_labels = self.data[sample_id]['labels']

        labels = {}
        for task in self.task_specs:
            if task['type'] == MULTILABEL_BINARY:
                self._require_columns(sample_labels, task['columns'], task['name'])
                values = [_safe_int(sample_labels[col]) for col in task['columns']]
                labels[task['name']] = torch.tensor(values, dtype=torch.int64)
            elif task['type'] == MULTICLASS:
                self._require_columns(sample_labels, [task['column']], task['name'])
                labels[task['name']] = torch.tensor(_safe_int(sample_labels[task['column']]), dtype=torch.int64)
            elif task['type'] == SURVIVAL:
                self._require_columns(sample_labels, [task['event_col'], task['bin_col'], task['time_col']], task['name'])
                labels[task['event_col']] = torch.tensor(_safe_int(sample_labels[task['event_col']]), dtype=torch.int64)
                labels[task['bin_col']] = torch.tensor(_safe_int(sample_labels[task['bin_col']]), dtype=torch.int64)
                labels[task['time_col']] = torch.tensor(_safe_int(sample_labels[task['time_col']]), dtype=torch.int64)

        if self.extra_attrs:
            labels['extra_attrs'] = {}
            for attr in self.extra_attrs:
                labels['extra_attrs'][attr] = torch.tensor(self.data[sample_id][attr][0])

        labels['id'] = sample_id
        return labels

    @staticmethod
    def _require_columns(sample_labels, columns, task_name):
        missing = [col for col in columns if col not in sample_labels]
        if missing:
            raise KeyError(f"Missing label column(s) {missing} for task {task_name}.")

    @staticmethod
    def _legacy_task_specs(task_names):
        if not task_names:
            raise ValueError("LabelDataset_multi_target requires task_specs or legacy task_names.")
        return [
            {
                "name": "biomarkers",
                "type": MULTILABEL_BINARY,
                "columns": task_names[0],
                "missing_label": 2,
            },
            {
                "name": "survival",
                "type": SURVIVAL,
                "event_col": "survival_event",
                "bin_col": "survival_bins",
                "time_col": "survival_days",
                "missing_event": 2,
                # Raw survival_bins use 0-3 for censored samples and 4-7 for event samples.
                # The model predicts four time-bin hazards; loss receives raw_bin % 4.
                "raw_label_classes": 8,
                "time_bins": 4,
            },
        ]
            
        
