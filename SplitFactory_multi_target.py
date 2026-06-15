import os
import yaml

from datasets.DataSplit import DataSplit


class SplitFactory_multi_target:    
    @staticmethod
    def from_local(path_to_split, path_to_config):
        '''
        Returns the datasplit from a local path.
        
        Args:
            path_to_split (str): Path to the split
            path_to_config (str): Path to the task config file
            
        Returns:
            split (DataSplit): Split object
            task_info (dict): Task metadata
        '''
        assert os.path.exists(path_to_split), f"Path to split {path_to_split} does not exist locally."
        assert os.path.exists(path_to_config), f"Path to split config {path_to_config} does not exist locally."
        
        task_info = SplitFactory_multi_target.get_task_info(path_to_config=path_to_config)
        target_cols = task_info['target_cols']          # list[str]
        label_cols = target_cols                           # ✅ 多列标签
        split = DataSplit(path = path_to_split,
                        id_col = task_info['sample_col'],
                        attr_cols = task_info['extra_cols'] + ['slide_id'],
                        label_cols = label_cols
                        # label_cols = [task_info['task_col']]
                        )
        return split, task_info
    
    @staticmethod
    def from_hf(saveto, source, task):
        raise NotImplementedError("This simplified project only supports local split/config files.")
    
    @staticmethod
    def get_task_info(path_to_config = None,
                      saveto = None,
                      source = None,
                      task = None):
        '''
        Returns the task metadata for a given source and task.
        
        Args:
            path_to_config (str): Path to the task config file. If None, will download the config file from HuggingFace.
            saveto (str): Path to save the split. Required if path_to_config is None.
            source (str): Name of source dataset. Required if path_to_config is None.
            task (str): Name of task. Required if path_to_config is None.
            
        Returns:
            task_info (dict): Task metadata
        '''
        if path_to_config is None:
            assert saveto is not None, "saveto must be provided if path_to_config is None."
            assert source is not None, "source must be provided if path_to_config is None."
            assert task is not None, "task must be provided if path_to_config is None."
            _, path_to_config = SplitFactory_multi_target.from_hf(saveto, source, task)
            
        with open(path_to_config, 'r') as task_info:
            task_info = yaml.safe_load(task_info)
            
        # Check that task_info has the required format
        assert 'sample_col' in task_info and isinstance(task_info['sample_col'], str), f"sample_col (str) not found in task config at {path_to_config}."
        assert 'extra_cols' in task_info and isinstance(task_info['extra_cols'], list), f"extra_cols (list[str]) not found in task config at {path_to_config}."
        # assert 'label_dict' in task_info and isinstance(task_info['label_dict'], dict), f"label_dict (dict) not found in task config at {path_to_config}."
        assert 'target_cols' in task_info and isinstance(task_info['target_cols'], list), \
            f"'target_cols' (list[str]) not found in task config at {path_to_config}."
        assert all(isinstance(c, str) and len(c) > 0 for c in task_info['target_cols']), \
            f"'target_cols' must be a non-empty list[str], got: {task_info['target_cols']}"
        assert 'metrics' in task_info and isinstance(task_info['metrics'], list), f"metrics (list[str]) not found in task config at {path_to_config}."
            
        return task_info
