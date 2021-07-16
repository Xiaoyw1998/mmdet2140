from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class FanGuangyiDataset(CustomDataset):
    # CLASSES = ('clothes', 'no_clothes', 'person_clothes', 'person_no_clothes')
    CLASSES = ('clothes', 'person')

    def load_annotations(self, ann_file):
        import pickle
        with open(ann_file, 'rb') as f:
            data_infos = pickle.load(f)
        return data_infos
