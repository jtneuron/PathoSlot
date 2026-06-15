from datasets.CombinedDataset import CombinedDataset
from datasets.LabelDataset_multi_target import LabelDataset_multi_target
from datasets.MultimodalSlideDataset import MultimodalSlideDataset

"""
Dataset construction for the MultimodalSlotModel path.
"""

class DatasetFactory:

    @staticmethod
    def from_slide_embeddings(**kwargs):
        '''
        Creates a dataset that returns precomputed multimodal slide embeddings and labels.
        '''
        return CombinedDataset({
            'slide': DatasetFactory._slide_embeddings_dataset(**kwargs),
            'labels': DatasetFactory._labels_dataset(kwargs['split'], kwargs['task_name'])
        })

    @staticmethod
    def _slide_embeddings_dataset(split,
                                  pooled_embeddings_dir=None,
                                  **kwargs):
        '''
        Creates a dataset that loads h5 files with vis_features/text_features/name_embedding.
        '''
        return MultimodalSlideDataset(split, load_from=pooled_embeddings_dir)

    @staticmethod
    def _labels_dataset(split, task):
        '''
        Creates a dataset that loads sample labels.
        
        Args:
            split (Split): Split object
            task (str): Name of the task
        '''
        return LabelDataset_multi_target(split, task_names=[task], dtype='int')
