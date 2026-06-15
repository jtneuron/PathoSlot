import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from itertools import product

import numpy as np
import torch
from torch import nn

from DatasetFactory import DatasetFactory
from MultimodalSlotModel import MultimodalSlotModel
from SplitFactory_multi_target import SplitFactory_multi_target
from experiments.FinetuningExperiment_multi_target import FinetuningExperiment_multi_target
from helpers.GPUManager import GPUManager
from optim.NLLSurvLoss import NLLSurvLoss


COMBINE_TRAIN_VAL = False


class ExperimentFactory:
    """Factory for the only supported experiment: MultimodalSlotModel finetuning."""

    @staticmethod
    def finetune_multi_target_slot(
        split: str,
        task_config: str,
        pooled_embeddings_dir: str,
        saveto: str,
        base_learning_rate: float = 3e-4,
        gradient_accumulation: int = 1,
        weight_decay: float = 1e-5,
        num_epochs: int = 1,
        scheduler_type: str = "cosine",
        optimizer_type: str = "AdamW",
        balanced: bool = True,
        save_which_checkpoints: str = "last-1",
        gpu: int = -1,
        batch_size: int = 1,
        num_bootstraps: int = 100,
        eval_checkpoint: str = "best",
        view_progress: str = "bar",
        biomarker_loss_weights=None,
    ):
        assert batch_size == 1, "Only batch_size=1 is supported by the current slot pipeline."

        split_obj, task_info, dataset = ExperimentFactory._prepare_internal_dataset(
            split_path=split,
            task_config=task_config,
            saveto=saveto,
            pooled_embeddings_dir=pooled_embeddings_dir,
            combine_train_val=COMBINE_TRAIN_VAL,
        )

        biomarkers = task_info["biomarker"]
        num_biomarkers = len(biomarkers)

        if balanced:
            cls_loss = build_balanced_multilabel_loss_per_fold(split_obj, biomarkers)
        else:
            cls_loss = nn.BCEWithLogitsLoss(reduction="none")

        loss = [
            cls_loss,
            NLLSurvLoss(alpha=0.0, eps=1e-7, reduction="mean"),
        ]

        model_kwargs = {
            "slide_encoder": None,
            "post_pooling_dim": 768,
            "task_name": "biomarkers",
            "num_classes": num_biomarkers,
            "loss": loss,
            "biomarker_weights": biomarker_loss_weights,
        }

        scheduler_config = ExperimentFactory._scheduler_config(scheduler_type)
        optimizer_config = ExperimentFactory._optimizer_config(
            optimizer_type=optimizer_type,
            base_learning_rate=base_learning_rate,
            weight_decay=weight_decay,
        )

        device = ExperimentFactory._resolve_device(gpu)
        experiment = FinetuningExperiment_multi_target(
            dataset=dataset,
            batch_size=batch_size,
            model_constructor=MultimodalSlotModel,
            model_kwargs=model_kwargs,
            num_epochs=num_epochs,
            accumulation_steps=gradient_accumulation,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            save_which_checkpoints=save_which_checkpoints,
            num_bootstraps=num_bootstraps,
            precision=torch.float32,
            device=device,
            results_dir=saveto,
            view_progress=view_progress,
        )
        experiment.eval_checkpoint = eval_checkpoint
        return experiment

    @staticmethod
    def _prepare_internal_dataset(
        split_path: str,
        task_config: str,
        saveto: str,
        pooled_embeddings_dir: str,
        combine_train_val: bool,
    ):
        split, task_info = SplitFactory_multi_target.from_local(split_path, task_config)
        if combine_train_val:
            split.replace_folds("val", "train")
        split.save(os.path.join(saveto, "split.csv"), row_divisor="slide_id")

        dataset = DatasetFactory.from_slide_embeddings(
            split=split,
            task_name=task_info["biomarker"],
            pooled_embeddings_dir=pooled_embeddings_dir,
        )
        return split, task_info, dataset

    @staticmethod
    def _scheduler_config(scheduler_type: str):
        if scheduler_type == "cosine":
            return {
                "type": "cosine",
                "eta_min": 1e-8,
                "step_on": "accumulation-step",
            }
        raise NotImplementedError('Only scheduler_type="cosine" is supported in the slot pipeline.')

    @staticmethod
    def _optimizer_config(optimizer_type: str, base_learning_rate: float, weight_decay: float):
        if optimizer_type == "AdamW":
            return {
                "type": "AdamW",
                "base_lr": base_learning_rate,
                "weight_decay": weight_decay,
            }
        raise NotImplementedError('Only optimizer_type="AdamW" is supported in the slot pipeline.')

    @staticmethod
    def _resolve_device(gpu: int):
        if not torch.cuda.is_available():
            return "cpu"
        return f"cuda:{gpu if gpu != -1 else GPUManager.get_best_gpu(min_mb=500)}"


def parse_task_code(task_code):
    data_source, task_name = task_code.split("--")
    if "==" in data_source:
        train_source, test_source = data_source.split("==")
        if train_source == test_source:
            raise ValueError(f"train_source and test_source must differ in task_code={task_code}")
        return train_source, test_source, task_name
    return data_source, None, task_name


def generate_exp_id(hyperparams):
    return "_".join(sorted([f"{k}={v}" for k, v in hyperparams.items()]))


def generate_arg_combinations(variables):
    variables = {k.lower(): make_list(v) for k, v in variables.items()}
    return [dict(zip(variables.keys(), combination)) for combination in product(*variables.values())]


def make_list(x):
    return x if isinstance(x, list) else [x]


def _safe_label(value, missing_value=2):
    try:
        if value is None:
            return missing_value
        if isinstance(value, str) and value.strip() == "":
            return missing_value
        return int(float(value))
    except Exception:
        return missing_value


def build_balanced_multilabel_loss_per_fold(split, task_names: list):
    loss_dict = {}
    for fold in range(split.num_folds):
        rows = [
            sample
            for sample in split.data
            if sample.get("folds") is not None and sample["folds"][fold] == "train"
        ]
        y = np.array(
            [[_safe_label(sample["labels"].get(task_name)) for task_name in task_names] for sample in rows],
            dtype=np.int64,
        )
        if y.ndim != 2:
            raise ValueError(f"Expected multilabel y with shape [N,C], got {y.shape}.")

        mask = y <= 1
        pos = ((y == 1) & mask).sum(axis=0).astype(np.float32)
        neg = ((y == 0) & mask).sum(axis=0).astype(np.float32)

        pos_weight = np.ones_like(pos, dtype=np.float32)
        valid = pos > 0
        pos_weight[valid] = neg[valid] / pos[valid]

        loss_dict[fold] = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=torch.from_numpy(pos_weight).float(),
        )
    return loss_dict
