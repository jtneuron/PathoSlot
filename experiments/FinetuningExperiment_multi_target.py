import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import numpy as np
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
import json
import warnings
import shutil
import pandas as pd

# Import optimizers
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import AdamW

# Other imports
from datasets.BaseDataset import BaseDataset
from experiments.BaseExperiment import BaseExperiment
from experiments.utils.LoggingMixin import LoggingMixin
from experiments.utils.ClassificationMixin import ClassificationMixin
from experiments.utils.SurvivalMixin import SurvivalMixin


# Turn off tokenizer parallelism to avoid warnings from dataloader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
This file contains the FinetuningExperiment class, which is used to train and test supervised neural network models.
"""

class FinetuningExperiment_multi_target(LoggingMixin, ClassificationMixin, SurvivalMixin, BaseExperiment):
    def __init__(self,
                 dataset: BaseDataset,
                 batch_size: int,
                 model_constructor: callable,
                 model_kwargs: dict,
                 num_epochs: int,
                 accumulation_steps: int,
                 optimizer_config: dict,
                 scheduler_config: dict,
                 save_which_checkpoints: str,
                 num_bootstraps: int,
                 precision: torch.dtype,
                 device: str,
                 results_dir: str,
                 view_progress: str = 'bar',
                 lr_logging_interval: int = None,
                 seed: int = 7,
                 **kwargs):
        """
        Base class for all experiments.

        Args:
            dataset (BaseDataset): Dataset object
            batch_size (int): Batch size.
            model_constructor (callable): Model class which can be called to create model instance.
            model_kwargs: Arguments passed to model_constructor.
            num_epochs (int): Number of epochs.
            accumulation_steps (int): Number of batches to accumulate gradients over before stepping optimizer.
            optimizer_config: Optimizer config.
            scheduler_config: LR scheduler config.
            save_which_checkpoints (str): Mode of saving checkpoints.
            num_bootstraps (int): Number of bootstraps to use for computing 95% CI.
            precision (torch.dtype): Precision to use for training.
            device (str): Device to use for training.
            results_dir (str): Where to save results.
            view_progress (str, optional): How to log progress. Can be 'bar' or 'verbose'. Defaults to 'bar'.
            lr_logging_interval (int, optional): Interval at which to log learning rate to dashboard (in number of accumulation steps). Defaults to None (do not log).
            seed (int): Seed for reproducibility.
            **kwargs: Additional arguments to save in config.json
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.model_constructor = model_constructor
        self.model_kwargs = model_kwargs
        self.num_epochs = num_epochs
        self.accumulation_steps = accumulation_steps
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.save_which_checkpoints = save_which_checkpoints
        self.num_bootstraps = num_bootstraps
        self.precision = precision
        self.device = device
        self.results_dir = results_dir
        self.view_progress = view_progress
        self.lr_logging_interval = lr_logging_interval
        self.seed = seed
        self.set_seed(self.seed)
        # Evaluation checkpoint mode: "best" prefers best.pt; "latest" uses latest.pt / latest epoch.
        self.eval_checkpoint = getattr(self, "eval_checkpoint", "best")
        
        # Set kwargs as extra attributes for saving in config.json
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Ensure that val set is nonempty if save_which_checkpoints is 'best-val-loss'
        if self.save_which_checkpoints == 'best-val-loss':
            assert self.dataset.get_subset(iteration = 0, fold = 'val') is not None, "Split must contain validation samples if save_which_checkpoints is 'best-val-loss'."

    def train(self):
        '''
        Runs training (and optionally validation) epochs for all folds of the experiment.
        '''
        print(f'\nExperiment dir: {self.results_dir}')
        self.save_config(os.path.join(self.results_dir, 'config.json'))
        self.train_results_dir = self.results_dir  # Store a copy of the training results dir for loading model in self.test(), in case want to save test results in a different directory

        ### Loop through folds
        for self.current_iter in range(self.dataset.num_folds):
            
            print("############################################################################################################")
            print(f"Training: Fold {self.current_iter + 1} of {self.dataset.num_folds}...")
            self.loggers = self.init_loggers(save_dir = os.path.join(self.results_dir, 'training_metrics', f'fold_{self.current_iter}'))

            ### Initialize train and val dataloaders
            self.dataloaders = {mode: self.dataset.get_dataloader(self.current_iter, mode, batch_size=self.batch_size) for mode in ['train', 'val']}
            
            ### Initialize model
            self.model = self.model_constructor(**self.model_kwargs, device = self.device)
            self.save_model_architecture(self.model, os.path.join(self.results_dir, f'model.txt'))
            
            ### Initialize optimizer and scheduler
            self.optimizer = self._init_optimizer()
            self.scheduler = self._init_scheduler()

            ### Prepare grad scaler
            # Only use GradScaler for FP16 training. bfloat16 does not require GradScaler: https://discuss.pytorch.org/t/bfloat16-training-explicit-cast-vs-autocast/202618/8
            try:
                self.grad_scaler = torch.amp.GradScaler('cuda', enabled = (self.precision == torch.float16)) 
            except:
                # Legacy (torch 2.0.0) implementation for compatibility with Gigapath
                self.grad_scaler = torch.cuda.amp.GradScaler(enabled = (self.precision == torch.float16))

            ### Initialize best loss and rank
            self.best_val_loss = 1e4            # Initialize to large number
            self.best_smooth_rank = 0           # Initialize to 0

            ### Prepare epoch loop
            if self.view_progress == 'bar':
                self.loop = tqdm(range(self.num_epochs))
            elif self.view_progress == 'verbose':
                self.loop = range(self.num_epochs)
            else:
                raise ValueError(f"view_progress must be 'bar' or 'verbose', got {self.view_progress} instead.")
            
            ### Loop through epochs
            for self.current_epoch in self.loop:
                for self.mode in ['train', 'val']:
                    if self.dataloaders[self.mode] is not None:
                        if self.view_progress == 'bar':
                            self.loop.set_description(f'      Epoch {self.current_epoch} {self.mode}')

                        start = time.time()
                        self._run_single_epoch()
                        end = time.time()

                        # Save progress to file
                        with open(os.path.join(self.results_dir, 'progress.txt'), 'a') as f:
                            f.write(f'DONE: Fold {self.current_iter + 1}/{self.dataset.num_folds} | {self.mode} | Epoch {self.current_epoch}\n')
                            
                        if self.view_progress == 'verbose':
                            print(f"Finished epoch = {self.current_epoch} in {end - start:.2f} seconds")
                            
        # After we finish all folds, try running a final "validation metrics" pass
        self.validate()

    def test(self):
        '''
        Evaluate the model on the test set for each fold.
        '''
        self._eval(split='test')

    def validate(self):
        """
        Evaluate the model on the validation set for each fold.
        """
        self._eval(split='val')
        
    def _eval(self, split: str):
        """
        多任务版评估：
        - 内部循环所有 fold，不需要外部指定 fold。
        - 每个 fold 保存 per-sample prediction、per-fold metrics、ROC/PR 原始曲线数据。
        - 所有 fold 汇总保存：
            1) {split}_metrics_summary.json：原有 bootstrap 95%CI 汇总；
            2) {split}_fold_metrics_raw.csv：每折原始指标；
            3) {split}_mean_sd_summary.csv：论文用 Mean(SD) / mean ± sd。
        - 不在这里做模型间显著性检验；显著性后续用多个模型的 fold_metrics_raw.csv 单独计算。
        """
        outputs_across_folds = []
        fold_metrics_across_folds = []

        loop = tqdm(range(self.dataset.num_folds))
        for self.current_iter in loop:
            # ---- dataloader ----
            eval_dataloader = self.dataset.get_dataloader(self.current_iter, split, batch_size=1)
            if eval_dataloader is None:
                return

            loop.set_description(
                f'Running {split} split on {len(eval_dataloader.dataset)} samples (fold {self.current_iter})'
            )

            # ---- checkpoint: best/latest 可切换 ----
            checkpoint_dir = os.path.join(self.results_dir, 'checkpoints', f'fold_{self.current_iter}')
            ckpt_path = self._pick_checkpoint(
                checkpoint_dir,
                which=getattr(self, "eval_checkpoint", "best")
            )

            # ---- load & freeze model ----
            model = self.model_constructor(**self.model_kwargs, device=self.device)
            model = self.load_checkpoint(model, ckpt_path)
            model = self.freeze(model)
            model.eval()

            # ---- accumulate 多任务预测 ----
            fold_output = self._accumulate_preds(eval_dataloader, model)

            # ---- 保存所有 biomarker 的 per-sample 预测 ----
            self._save_biomarker_predictions(split, self.current_iter, fold_output)

            # ---- 保存 survival per-sample 预测；后续画 KM 曲线需要 id/event/time/risk ----
            self._save_survival_predictions(split, self.current_iter, fold_output)

            # ---- per-fold metrics & 图 & ROC/PR raw curve data ----
            per_fold_save_dir = os.path.join(
                self.results_dir, f'{split}_metrics', f'fold_{self.current_iter}'
            )
            os.makedirs(per_fold_save_dir, exist_ok=True)
            fold_metrics = self._compute_metrics(fold_output, per_fold_save_dir)

            outputs_across_folds.append(fold_output)
            fold_metrics_across_folds.append(fold_metrics)

        # ---- 原有 bootstrap 95%CI 汇总，保留 ----
        summary = self._finalize_multi_task_metrics(split, outputs_across_folds)

        with open(os.path.join(self.results_dir, f'{split}_metrics_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

        # ---- 新增：fold-level raw metrics + Mean(SD) summary ----
        self._save_fold_metrics_raw_and_mean_sd(split, fold_metrics_across_folds)

    def _filter_survival(self, event, time, risk):
        """
        Survival 有效样本过滤，统一用于 per-fold metrics 和 bootstrap summary。
        约定：
          - event 只允许 0/1；
          - time 必须有限且 > 0；
          - risk 必须有限。
        """
        event = np.asarray(event).reshape(-1)
        time = np.asarray(time).reshape(-1)
        risk = np.asarray(risk).reshape(-1)

        n = min(len(event), len(time), len(risk))
        event, time, risk = event[:n], time[:n], risk[:n]

        m = np.isin(event, [0, 1])
        m = m & np.isfinite(time) & (time > 0)
        m = m & np.isfinite(risk)

        return event[m].astype(int), time[m].astype(float), risk[m].astype(float)

    def _finalize_multi_task_metrics(self, split, outputs_across_folds):
        """
        多任务 metrics 汇总：
          - biomarker: 对每个 biomarker 通道单独做二分类评估，
                       每个 biomarker 自己 bootstrap + classification_metrics + get_95_ci
          - survival : 使用 survival_metrics + bootstrap + get_95_ci

        参数:
            split: 'val' 或 'test'
            outputs_across_folds: list[dict]，每个元素是 _accumulate_preds 的返回：
                {
                    "biomarker": {"labels":[Ni,C], "preds":[Ni,C]},
                    "survival":  {"event":[Ni], "time":[Ni], "risk":[Ni]}
                }

        返回:
            summary = {
                "biomarker": {
                    "per_marker": {
                        "marker_0": {...95% CI...},
                        "marker_1": {...},
                        ...
                    }
                },
                "survival":  <95% CI 字典>
            }
        """
        if len(outputs_across_folds) == 0:
            return {}

        # =============================
        # 1. 准备 biomarker 的 label/pred 列表（按 fold）
        # =============================
        bio_labels_across_folds = [o["biomarker"]["labels"] for o in outputs_across_folds]  # list of [Ni, C]
        bio_preds_across_folds  = [o["biomarker"]["preds"]  for o in outputs_across_folds]  # list of [Ni, C]

        # 维度检查 & 通道数
        num_biomarkers = bio_labels_across_folds[0].shape[1]
        assert all(lbl.shape[1] == num_biomarkers for lbl in bio_labels_across_folds), \
            "所有 fold 的 biomarker 通道数必须一致"

        # =============================
        # 2. 准备 survival 的 label/pred 列表（按 fold）
        #    labels 需要是 dict，和原先 survival 分支一样
        # =============================
        surv_labels_across_folds = []
        surv_preds_across_folds  = []

        for o in outputs_across_folds:
            e, t, r = self._filter_survival(
                o["survival"]["event"],
                o["survival"]["time"],
                o["survival"]["risk"]
            )
            if e.size == 0:
                continue

            surv_labels_across_folds.append(np.stack([e, t], axis=1))  # [N,2]
            surv_preds_across_folds.append(r)

        # for o in outputs_across_folds:
        #     surv_labels_across_folds.append({
        #         'survival_event': o["survival"]["event"],
        #         'survival_days':  o["survival"]["time"],
        #     })
        #     surv_preds_across_folds.append(o["survival"]["risk"])
            
        # =============================
        # 3. biomarker：对每个通道单独做 bootstrap + classification_metrics
        # =============================
        folder_root = os.path.join(self.results_dir, f"{split}_metrics")
        os.makedirs(folder_root, exist_ok=True)

        folder_bio = os.path.join(folder_root, "biomarker")
        os.makedirs(folder_bio, exist_ok=True)

        biomarker_ci = {}  # 存每个 marker 的 95% CI

        for marker_idx in range(num_biomarkers):
            marker_name = f"marker_{marker_idx}"

            # ---- 3.1 按 fold 拆出当前通道的 labels & preds (prob of positive) ----
            labels_folds_j = [lbl[:, marker_idx] for lbl in bio_labels_across_folds]  # list of [Ni]
            preds_folds_j  = [prd[:, marker_idx] for prd in bio_preds_across_folds]   # list of [Ni]

            # ---- 3.2 对当前 biomarker 做 bootstrap ----
            bootstraps_j = self.bootstrap(
                labels_folds_j,
                preds_folds_j,
                self.num_bootstraps
            )

            scores_across_bootstraps_j = []

            marker_folder = os.path.join(folder_bio, marker_name)
            os.makedirs(marker_folder, exist_ok=True)

            from tqdm import tqdm as _tqdm
            for b_idx, (labels_arr, preds_pos_arr) in enumerate(
                _tqdm(bootstraps_j, desc=f'Computing {self.num_bootstraps} bootstraps for {marker_name}')
            ):
                labels_arr = np.asarray(labels_arr)
                preds_pos_arr = np.asarray(preds_pos_arr)

                # 只保留 0/1，>1 视为未知，不参与该 marker 评估
                valid_mask = np.isin(labels_arr, [0, 1])
                y_bin = labels_arr[valid_mask]
                p_pos = preds_pos_arr[valid_mask]

                if y_bin.size == 0:
                    # 整个 bootstrap 样本没有有效标签，跳过
                    continue

                # 构造二分类 [N,2] 概率：neg = 1 - pos, pos = p_pos
                preds_2col = np.stack([1.0 - p_pos, p_pos], axis=1)  # [N_valid, 2]

                # 调用原来的 classification_metrics，当成一个二分类任务
                # 这里不给 saveto，每个 bootstrap 单独保存：
                scores_dict = self.classification_metrics(
                    y_true=y_bin,
                    preds=preds_2col,
                    num_classes=2,
                    threshold=None,
                    saveto=None
                )
                # 只保留 overall 部分参与 CI 统计
                scores_across_bootstraps_j.append(scores_dict["overall"])

                # 单个 bootstrap 的 metrics.json 也按原习惯写出去
                folder_curr = os.path.join(marker_folder, f"bootstrap_{b_idx}")
                os.makedirs(folder_curr, exist_ok=True)
                file_path = os.path.join(folder_curr, "metrics.json")
                with open(file_path, "w") as f:
                    json.dump(scores_dict, f, indent=4)

            if len(scores_across_bootstraps_j) == 0:
                # 全部 bootstrap 都没有有效样本
                biomarker_ci[marker_name] = {}
            else:
                # 对当前 biomarker 的 bootstrap 结果做 95% CI
                biomarker_ci[marker_name] = self.get_95_ci(scores_across_bootstraps_j)

        # =============================
        # 4. survival：同理，走原 survival 分支的 bootstrap 逻辑
        # =============================
        bootstraps_surv = self.bootstrap(
            surv_labels_across_folds,
            surv_preds_across_folds,
            self.num_bootstraps
        )

        surv_scores_across_bootstraps = []

        folder_surv = os.path.join(folder_root, "survival")
        os.makedirs(folder_surv, exist_ok=True)

        from tqdm import tqdm as _tqdm2
        for idx, (labels, preds) in enumerate(
            _tqdm2(bootstraps_surv, desc=f'Computing {self.num_bootstraps} survival bootstraps')
        ):
            labels = np.asarray(labels)
            if labels.ndim == 1:
                labels = labels.reshape(1, -1)
            e_raw = labels[:, 0]
            t_raw = labels[:, 1]
            e, t, r = self._filter_survival(e_raw, t_raw, preds)
            # e, t, r = self._filter_survival(labels['survival_event'], labels['survival_days'], preds)
            if e.size == 0:
                scores_dict = {}
            else:
                scores_dict = self.survival_metrics(e, t, r)
                
            surv_scores_across_bootstraps.append(scores_dict)

            folder_curr = os.path.join(folder_surv, f"bootstrap_{idx}")
            os.makedirs(folder_curr, exist_ok=True)
            file_path = os.path.join(folder_curr, "metrics.json")
            with open(file_path, "w") as f:
                json.dump(scores_dict, f, indent=4)

        nonempty_surv_scores = [score for score in surv_scores_across_bootstraps if score]
        surv_summary = self.get_95_ci(nonempty_surv_scores) if nonempty_surv_scores else {}

        # =============================
        # 5. 打包总 summary，写到 {split}_metrics_summary.json
        # =============================
        summary = {
            "biomarker": {
                "per_marker": biomarker_ci   # 不假设 biomarker 个数，也不写死名字
            },
            "survival": surv_summary
        }
        return summary

    def _accumulate_preds(self, dataloader, model):
        """
        多任务：一次 forward 得到 biomarker + survival 的全部预测
        输出结构：
        {
            "biomarker": {"labels": [...], "preds": [...]},
            "survival":  {"event": [...], "time": [...], "risk": [...]}
        }
        """
        bio_ids_all = []          # ✅ 新增
        bio_labels_all = []
        bio_preds_all = []
        surv_event_all = []
        surv_time_all = []
        surv_risk_all = []

        device_type = self.device.split(':')[0]
        with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=self.precision, enabled=self.precision != torch.float32):
            for batch in dataloader:

                # =====================
                # 1. Forward
                # =====================
                logits = model(batch, output='logits')

                biomarker_logits = logits['biomarker'][0]      # shape [3]
                survival_logits  = logits['survival'][0]       # shape [4 time bins]

                # =====================
                # 2. Biomarker 预测（多标签 → Sigmoid）
                # =====================
                bio_prob = torch.sigmoid(biomarker_logits).cpu().numpy()

                # ✅ 取 sample id（batch_size=1）
                # 推荐写得鲁棒点：支持 list / tensor / str
                sid = batch.get('ids', None)
                if sid is None:
                    assert 0
                # batch_size=1 → 取第一个
                if isinstance(sid, (list, tuple)):
                    sid0 = sid[0]
                elif hasattr(sid, 'tolist'):  # tensor
                    sid0 = sid.tolist()[0]
                else:
                    sid0 = sid
                bio_ids_all.append(str(sid0))

                # biomarker label
                bio_label = batch['labels']['biomarkers'].cpu().numpy()[0]

                bio_labels_all.append(bio_label)   # [3]
                bio_preds_all.append(bio_prob)     # [3]

                # =====================
                # 3. Survival 预测（NLLSurvLoss → risk score）
                # =====================
                risk = float(self._calculate_risk(survival_logits).detach().cpu().item())

                event = batch['labels']['survival_event'].cpu().numpy()[0]
                time  = batch['labels']['survival_days'].cpu().numpy()[0]

                surv_event_all.append(event)
                surv_time_all.append(time)
                surv_risk_all.append(risk)

        # =========================
        # 打包多任务预测结果
        # =========================
        output = {
            "biomarker": {
                "ids":    np.array(bio_ids_all),         # ✅ 新增
                "labels": np.array(bio_labels_all),   # shape [N,num_biomakers]
                "preds":  np.array(bio_preds_all),    # shape [N,num_biomakers]
            },
            "survival": {
                "ids":   np.array(bio_ids_all),       # shape [N]
                "event": np.array(surv_event_all),    # shape [N]
                "time":  np.array(surv_time_all),     # shape [N]
                "risk":  np.array(surv_risk_all),     # shape [N]
            }
        }
        return output
    
    def _compute_metrics(self, fold_output, save_dir):
        """
        多任务版本的 per-fold metrics 计算与落盘。
        这里不再区分 task_type，默认为：
            fold_output (来自 _accumulate_preds 的 dict)

        保存内容：
            save_dir/
            biomarker/
                marker_0/  (该 marker 的 ROC/PR/CM 图和 metrics.json)
                marker_1/
                ...
            survival/
                metrics.json
        返回：
            一个简单 dict，给你想用 fold 级别分数的时候参考用；最终汇总还是交给 `_finalize_multi_task_metrics`。
        """

        # ================== biomarker ==================
        bio_labels = np.asarray(fold_output["biomarker"]["labels"])  # [N, C]
        bio_preds  = np.asarray(fold_output["biomarker"]["preds"])   # [N, C]
        num_markers = bio_labels.shape[1]

        biomarker_root = os.path.join(save_dir, "biomarker")
        os.makedirs(biomarker_root, exist_ok=True)

        per_marker_scores = {}

        for j in range(num_markers):
            marker_name = f"marker_{j}"
            marker_dir = os.path.join(biomarker_root, marker_name)
            os.makedirs(marker_dir, exist_ok=True)

            yj = bio_labels[:, j]    # [N]
            pj = bio_preds[:, j]     # [N]，sigmoid 之后的 P(pos)

            # 只保留 label <=1 的样本；>1 当作未知，不参与指标
            valid = np.isin(yj, [0, 1])
            yj = yj[valid].astype(int)
            pj = pj[valid]

            if yj.size == 0:
                # 这个 marker 在本 fold 没有有效标签
                per_marker_scores[marker_name] = {}
                continue

            # 构造二分类概率 [N,2]：第一列 P(neg)，第二列 P(pos)
            preds_2col = np.stack([1.0 - pj, pj], axis=1)

            # 保存 ROC / PR 原始绘图数据，方便后续统一重画
            self._save_binary_curve_data(
                y_true=yj,
                preds_2col=preds_2col,
                save_dir=marker_dir
            )

            # 画图
            self.auc_roc(
                y_true=yj,
                preds=preds_2col,
                num_classes=2,
                saveto=os.path.join(marker_dir, "roc_curves.png")
            )

            self.precision_recall(
                y_true=yj,
                preds=preds_2col,
                num_classes=2,
                saveto=os.path.join(marker_dir, "pr_curves.png")
            )

            self.confusion_matrix(
                y_true=yj,
                preds=preds_2col,
                num_classes=2,
                saveto=os.path.join(marker_dir, "confusion_matrices.png")
            )

            # 数值指标
            scores = self.classification_metrics(
                y_true=yj,
                preds=preds_2col,
                num_classes=2,
                saveto=os.path.join(marker_dir, "metrics.json")
            )
            # 只留 overall，方便后面折腾
            per_marker_scores[marker_name] = scores["overall"]

        # ================== survival ==================
        surv_event = np.asarray(fold_output["survival"]["event"])
        surv_time  = np.asarray(fold_output["survival"]["time"])
        surv_risk  = np.asarray(fold_output["survival"]["risk"])

        survival_dir = os.path.join(save_dir, "survival")
        os.makedirs(survival_dir, exist_ok=True)
        
        e, t, r = self._filter_survival(surv_event, surv_time, surv_risk)
        if e.size == 0:
            surv_scores = {}
        else:
            surv_scores = self.survival_metrics(e, t, r, saveto=os.path.join(survival_dir, "metrics.json"))


        # 返回一个结构化 dict，万一你后面还想用 per-fold 的话
        metrics = {
            "biomarker": {
                "per_marker": per_marker_scores
            },
            "survival": surv_scores
        }
        return metrics

    def _save_biomarker_predictions(self, split, fold_idx, fold_output):
        """
        Save per-sample biomarker predictions for all marker channels.
        """
        save_pred_root = os.path.join(self.results_dir, f"{split}_biomarker_preds", f"fold_{fold_idx}")
        os.makedirs(save_pred_root, exist_ok=True)

        ids = fold_output["biomarker"]["ids"]
        labels = np.asarray(fold_output["biomarker"]["labels"])
        preds = np.asarray(fold_output["biomarker"]["preds"])

        data = {"id": ids}
        num_markers = preds.shape[1]
        for j in range(num_markers):
            data[f"marker_{j}_prob"] = preds[:, j]
            data[f"marker_{j}_pred"] = (preds[:, j] >= 0.5).astype(int)
            if labels.ndim == 2 and labels.shape[1] > j:
                data[f"marker_{j}_label"] = labels[:, j]

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(save_pred_root, "biomarker_preds.csv"), index=False)

        # 兼容原先 marker0_preds 路径，避免你旧脚本找不到文件
        if num_markers > 0:
            legacy_root = os.path.join(self.results_dir, f"{split}_marker0_preds", f"fold_{fold_idx}")
            os.makedirs(legacy_root, exist_ok=True)
            legacy_df = pd.DataFrame({
                "id": ids,
                "marker0_prob": preds[:, 0],
                "marker0_pred": (preds[:, 0] >= 0.5).astype(int),
                "marker0_label": labels[:, 0] if labels.ndim == 2 and labels.shape[1] > 0 else np.nan,
            })
            legacy_df.to_csv(os.path.join(legacy_root, "marker0_preds.csv"), index=False)

    def _save_survival_predictions(self, split, fold_idx, fold_output):
        """
        Save per-sample survival predictions. This is the required raw material for KM curves:
        id, event, time, risk.
        """
        save_pred_root = os.path.join(self.results_dir, f"{split}_survival_preds", f"fold_{fold_idx}")
        os.makedirs(save_pred_root, exist_ok=True)

        surv = fold_output["survival"]
        df = pd.DataFrame({
            "id": surv.get("ids", fold_output["biomarker"]["ids"]),
            "event": np.asarray(surv["event"]).reshape(-1),
            "time": np.asarray(surv["time"]).reshape(-1),
            "risk": np.asarray(surv["risk"]).reshape(-1),
        })
        df.to_csv(os.path.join(save_pred_root, "survival_preds.csv"), index=False)

    def _save_binary_curve_data(self, y_true, preds_2col, save_dir):
        """
        Save raw ROC/PR curve data for later re-plotting.
        """
        from sklearn.metrics import roc_curve, precision_recall_curve

        os.makedirs(save_dir, exist_ok=True)

        y_true = np.asarray(y_true).astype(int)
        p_pos = np.asarray(preds_2col)[:, 1]

        # ROC raw data
        try:
            fpr, tpr, roc_thresholds = roc_curve(y_true, p_pos)
            pd.DataFrame({
                "fpr": fpr,
                "tpr": tpr,
                "threshold": roc_thresholds,
            }).to_csv(os.path.join(save_dir, "roc_curve_raw.csv"), index=False)
        except Exception as e:
            warnings.warn(f"Failed to save ROC raw data: {e}")

        # PR raw data
        try:
            precision, recall, pr_thresholds = precision_recall_curve(y_true, p_pos)
            pr_thresholds_pad = np.append(pr_thresholds, np.nan)
            pd.DataFrame({
                "precision": precision,
                "recall": recall,
                "threshold": pr_thresholds_pad,
            }).to_csv(os.path.join(save_dir, "pr_curve_raw.csv"), index=False)
        except Exception as e:
            warnings.warn(f"Failed to save PR raw data: {e}")

    def _save_fold_metrics_raw_and_mean_sd(self, split, fold_metrics_across_folds):
        """
        Save:
          1. {split}_fold_metrics_raw.csv
          2. {split}_mean_sd_summary.csv

        这里只保存论文表格所需的 fold-level 原始指标和 Mean(SD)。
        不做 second-best、不做 paired t-test、不加星号。
        """
        rows = []

        for fold_idx, fold_metrics in enumerate(fold_metrics_across_folds):
            # biomarker metrics
            bio = fold_metrics.get("biomarker", {}).get("per_marker", {})
            for marker_name, metric_dict in bio.items():
                if not isinstance(metric_dict, dict):
                    continue
                for metric_name, value in metric_dict.items():
                    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                        rows.append({
                            "split": split,
                            "fold": fold_idx,
                            "task": "biomarker",
                            "target": marker_name,
                            "metric": metric_name,
                            "value": float(value),
                        })

            # survival metrics
            surv = fold_metrics.get("survival", {})
            if isinstance(surv, dict):
                for metric_name, value in surv.items():
                    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                        rows.append({
                            "split": split,
                            "fold": fold_idx,
                            "task": "survival",
                            "target": "survival",
                            "metric": metric_name,
                            "value": float(value),
                        })

        raw_df = pd.DataFrame(rows)
        raw_path = os.path.join(self.results_dir, f"{split}_fold_metrics_raw.csv")
        raw_df.to_csv(raw_path, index=False)

        if raw_df.empty:
            warnings.warn(f"No fold metrics found for {split}; skip mean±sd summary.")
            return

        summary_rows = []
        group_cols = ["split", "task", "target", "metric"]

        for keys, g in raw_df.groupby(group_cols):
            values = g["value"].astype(float).values
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

            summary_rows.append({
                "split": keys[0],
                "task": keys[1],
                "target": keys[2],
                "metric": keys[3],
                "n_folds": len(values),
                "mean": mean,
                "sd": sd,
                "mean_sd": f"{mean:.3f} ± {sd:.3f}",
                "mean_sd_parentheses": f"{mean:.3f} ({sd:.3f})",
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(self.results_dir, f"{split}_mean_sd_summary.csv")
        summary_df.to_csv(summary_path, index=False)

    def _pick_checkpoint(self, checkpoint_dir, which=None):
        """
        Pick checkpoint from a directory.

        which:
            - "best": prefer best.pt; if unavailable, fall back to latest.pt / latest epoch checkpoint.
            - "latest": prefer latest.pt; if unavailable, use latest epoch checkpoint.

        这样实现“最新 + 最好”：训练时会额外维护 latest.pt 和 best.pt；评估时可用
        self.eval_checkpoint = "best" 或 "latest" 控制读取哪个。
        """
        if which is None:
            which = getattr(self, "eval_checkpoint", "best")
        assert which in ["best", "latest"], f"which must be 'best' or 'latest', got {which}"

        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        best_path = os.path.join(checkpoint_dir, "best.pt")
        latest_path = os.path.join(checkpoint_dir, "latest.pt")

        if which == "best" and os.path.exists(best_path):
            return best_path
        if which == "latest" and os.path.exists(latest_path):
            return latest_path

        available_checkpoints = [
            f for f in os.listdir(checkpoint_dir)
            if f.endswith('.pt') and f not in ["best.pt", "latest.pt"]
        ]

        if not available_checkpoints:
            # 兜底：如果请求 best 但只有 latest，或请求 latest 但只有 best，也允许继续。
            if os.path.exists(latest_path):
                warnings.warn(f"Requested {which}, but only latest.pt is available in {checkpoint_dir}.")
                return latest_path
            if os.path.exists(best_path):
                warnings.warn(f"Requested {which}, but only best.pt is available in {checkpoint_dir}.")
                return best_path
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        def _epoch_key(filename):
            try:
                return int(filename.replace('.pt', '').split('_')[-1])
            except Exception:
                return -1

        latest = max(available_checkpoints, key=_epoch_key)
        if which == "best":
            warnings.warn(
                f"best.pt not found in {checkpoint_dir}. Using latest epoch checkpoint {latest} instead."
            )
        elif len(available_checkpoints) > 1:
            warnings.warn(
                f"{len(available_checkpoints)} checkpoints found in {checkpoint_dir}. "
                f"Using latest epoch checkpoint {latest}."
            )
        return os.path.join(checkpoint_dir, latest)

    def _run_single_epoch(self):
        """
        Runs a single training or validation epoch. After the epoch, the epoch metrics are stored in self.current_epoch_metrics.
        """

        # Set models to appropriate mode
        if self.mode == 'train':
            self.model.train()
            context_manager = torch.enable_grad()
        elif self.mode in ['val']:
            self.model.eval()
            context_manager = torch.inference_mode()
        else:
            raise ValueError('mode must be either "train", "val", or "test".')

        # Initialize performance trackers
        all_losses = []
        all_raw_losses = []
        all_info = []
        new_best_loss = False
        new_best_smooth_rank = False
        num_samples_processed = 0
        num_gradient_steps = 0
        optimizer_skipped = False

        # Loop over each batch in loader
        with context_manager:
            for batch_idx, batch in enumerate(self.dataloaders[self.mode]):
                num_samples_processed += len(batch['ids'])

                device_type = self.device.split(':')[0]
                with torch.autocast(device_type=device_type, dtype=self.precision, enabled=self.precision != torch.float32):
                    loss, info = self.model(batch, self.current_epoch, output='loss')
                    assert isinstance(loss, torch.Tensor), f"Loss must be a tensor, got {loss} instead"
                    assert isinstance(info, list), f"Info must be a list on CPU, got {info} instead"

                    raw_loss = loss
                    loss = loss / self.accumulation_steps

                # Update trackers
                all_losses.append(loss.cpu().detach().numpy())
                all_raw_losses.append(raw_loss.cpu().detach().numpy())
                all_info.extend(info)

                do_step = ((batch_idx + 1) % self.accumulation_steps == 0) or ((batch_idx + 1) == len(self.dataloaders[self.mode]))

                # Backward pass if training
                if self.mode == 'train':
                    self.grad_scaler.scale(loss).backward()
                    if do_step:
                        self.grad_scaler.step(self.optimizer)
                        current_scale = self.grad_scaler.get_scale()
                        self.grad_scaler.update()
                        optimizer_skipped = (self.grad_scaler.get_scale() < current_scale)
                        self.optimizer.zero_grad()
                        num_gradient_steps += 1

                        # Log learning rate to dashboard on every lr_logging_interval accumulation steps
                        if self.scheduler_config and self.lr_logging_interval is not None and num_gradient_steps % self.lr_logging_interval == 0:
                            self.log_lr(batch_idx + self.current_epoch * len(self.dataloaders[self.mode]))

                        # Update scheduler on accumulation step if step_on is 'accumulation-step'
                        if self.scheduler_config and self.scheduler_config['step_on'] == 'accumulation-step' and not optimizer_skipped:
                            try:
                                # API for custom LR scheduler
                                partial_epoch_progress = (batch_idx + 1) / len(self.dataloaders[self.mode])
                                self.scheduler.step(total_progress=(self.current_epoch + partial_epoch_progress) / self.num_epochs)
                            except:
                                try:
                                    # Default API for built-in LR schedulers
                                    self.scheduler.step()
                                except:
                                    raise Exception(f"Error stepping scheduler on accumulation-step.")
                else:
                    if do_step:
                        num_gradient_steps += 1

                # Update progress bar
                if self.view_progress == 'bar' and do_step:
                    denom = max(num_gradient_steps, 1)
                    self.loop.set_postfix(num_batches=f'{batch_idx + 1}/{len(self.dataloaders[self.mode])}',
                                          num_samples=num_samples_processed,
                                          avg_loss=f'{(sum(all_raw_losses)/len(all_raw_losses)):.4f}')

        # Update scheduler at end of epoch if step_on is 'epoch'
        if self.mode == 'train' and self.scheduler_config and self.scheduler_config['step_on'] == 'epoch' and not optimizer_skipped:
            self.scheduler.step()

        if len(all_losses) == 0:
            raise RuntimeError(f"No batches were processed in fold={self.current_iter}, mode={self.mode}.")

        # Save current epoch metrics
        self.current_epoch_metrics = {
            "loss": all_losses,
            "raw_loss": all_raw_losses,
            "info": all_info,
            'per_sample_loss': float(np.mean(all_raw_losses))
        }
        self.compute_extra_metrics()

        # Update best smooth rank
        if len(all_info) > 0 and isinstance(all_info[0], dict) and 'smooth_rank' in all_info[0].keys():
            smooth_rank = np.mean([info['smooth_rank'] for info in all_info])
            self.current_epoch_metrics['smooth_rank'] = smooth_rank
            if smooth_rank > self.best_smooth_rank:
                self.best_smooth_rank = smooth_rank
                new_best_smooth_rank = True
        else:
            assert self.save_which_checkpoints != 'best-smooth-rank', f"save_which_checkpoints cannot be 'best-smooth-rank' if smooth rank is not returned by the model."

        # Update best val loss using raw loss, not loss divided by accumulation_steps
        if self.mode == 'val':
            avg_loss = np.mean(all_raw_losses)
            if avg_loss < self.best_val_loss:
                self.best_val_loss = avg_loss
                new_best_loss = True

        # Save checkpoints
        save_conditions = [self.save_which_checkpoints == 'all',
                           self.save_which_checkpoints == 'best-val-loss' and new_best_loss,
                           self.save_which_checkpoints == 'best-smooth-rank' and new_best_smooth_rank,
                           self.save_which_checkpoints.startswith('every-') and (self.current_epoch + 1) % int(self.save_which_checkpoints.split('-')[1]) == 0,
                           self.save_which_checkpoints.startswith('last-') and (self.current_epoch + 1) > self.num_epochs - int(self.save_which_checkpoints.split('-')[1])]

        if any(save_conditions):
            ckpt_dir = os.path.join(self.results_dir, 'checkpoints', f'fold_{self.current_iter}')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{self.current_epoch}.pt")

            self.save_checkpoint(self.model, self.save_which_checkpoints, ckpt_path)

            # latest.pt：只要本轮保存了 checkpoint，就同步一份最新 checkpoint
            try:
                shutil.copy2(ckpt_path, os.path.join(ckpt_dir, "latest.pt"))
            except Exception as e:
                warnings.warn(f"Failed to copy latest checkpoint: {e}")

            # best.pt：val loss 最优时同步一份最佳 checkpoint
            if self.mode == "val" and new_best_loss:
                try:
                    shutil.copy2(ckpt_path, os.path.join(ckpt_dir, "best.pt"))
                except Exception as e:
                    warnings.warn(f"Failed to copy best checkpoint: {e}")

        self.log_loss(self.current_epoch)
        self.log_smooth_rank(self.current_epoch)

    def _init_scheduler(self):
        '''
        Returns a scheduler. Supports one of the built-in schedulers or a custom scheduler class.
        '''
        if isinstance(self.scheduler_config['type'], str):
            # Using built-in scheduler
            if self.scheduler_config['type'] == 'plateau':
                return torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode=self.scheduler_config['mode'],
                    factor=self.scheduler_config['factor'],
                    patience=self.scheduler_config['patience'],
                    verbose=True)
            elif self.scheduler_config['type'] == 'step':
                return torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=self.scheduler_config['step_size'],
                    gamma=self.scheduler_config['gamma'])
            elif self.scheduler_config['type'] == 'cosine':
                assert self.accumulation_steps == 1, "CosineAnnealingLR scheduler is not compatible with gradient accumulation."
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.num_epochs if self.scheduler_config['step_on'] == 'epoch' else len(self.dataloaders['train']) * self.num_epochs,
                    eta_min=self.scheduler_config['eta_min'])
            elif self.scheduler_config['type'] == 'cosine_warm_restart':
                return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=self.scheduler_config['T_0'],
                    T_mult=self.scheduler_config['T_mult'],
                    eta_min=self.scheduler_config['eta_min'])
            else:
                raise NotImplementedError(f"Scheduler type {self.scheduler_config['type']} not implemented.")

        elif callable(self.scheduler_config['type']):
            # Using custom scheduler class
            try:
                default_scheduler_args = {
                    'base_lr': self.optimizer_config['base_lr'],
                    'max_epochs': self.num_epochs,
                    'accumulation_steps': self.accumulation_steps,
                    'len_dataloader': len(self.dataloaders['train']),
                }

                return self.scheduler_config['type'](
                    optimizer=self.optimizer,
                    default_scheduler_args=default_scheduler_args,
                    custom_scheduler_args=self.scheduler_config)
            except Exception as e:
                raise Exception(f"Error initializing custom scheduler: {e}. \nExpected init format: CustomScheduler(optimizer: Optimizer, default_scheduler_args: dict, custom_scheduler_args: dict)")

        else:
            raise ValueError(f"Scheduler type must be a string or a callable, got {self.scheduler_config['type']} instead.")

    def _init_optimizer(self):
        '''
        Initialize optimizer.
        '''
        optimizer_type = self.optimizer_config['type']
        extra_kwargs = {k: v for k, v in self.optimizer_config.items() if k not in ['type', 'get_param_groups', 'param_group_args', 'base_lr']}

        if 'get_param_groups' in self.optimizer_config:
            param_groups = self.optimizer_config['get_param_groups'](self.model, **self.optimizer_config['param_group_args'])
            assert isinstance(param_groups, list), "get_param_groups must return a list of dictionaries."
            assert len(param_groups) > 0, "get_param_groups must return a non-empty list of dictionaries."
        else:
            param_groups = self.model.parameters()

        if optimizer_type.lower() == "adam":
            return Adam(param_groups, self.optimizer_config['base_lr'], **extra_kwargs)
        elif optimizer_type.lower() == 'sgd':
            return SGD(param_groups, self.optimizer_config['base_lr'], **extra_kwargs)
        elif optimizer_type.lower() == "adamw":
            return AdamW(param_groups, self.optimizer_config['base_lr'], **extra_kwargs)
        else:
            raise NotImplementedError(f"Optimizer {optimizer_type} not implemented.")
