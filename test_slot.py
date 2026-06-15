import os

from ExperimentFactory import ExperimentFactory


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    split = os.environ.get(
        "PATHOSLOT_SPLIT",
        os.path.join(PROJECT_ROOT, "data", "splits", "zhongshan_split.csv"),
    )
    task_config = os.environ.get(
        "PATHOSLOT_TASK_CONFIG",
        os.path.join(PROJECT_ROOT, "my_datatsets", "zhongshan_config.yaml"),
    )
    embeddings_dir = os.environ.get(
        "PATHOSLOT_EMBEDDINGS_DIR",
        os.path.join(PROJECT_ROOT, "data", "embeddings", "zhongshan"),
    )
    results_dir = os.environ.get(
        "PATHOSLOT_RESULTS_DIR",
        os.path.join(PROJECT_ROOT, "outputs", "slot_model"),
    )

    experiment = ExperimentFactory.finetune_multi_target_slot(
        split=split,
        task_config=task_config,
        pooled_embeddings_dir=embeddings_dir,
        saveto=results_dir,
        balanced=True,
        eval_checkpoint="latest",
    )
    experiment.test()


if __name__ == "__main__":
    main()
