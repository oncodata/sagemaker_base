import torch
import random
import torch.utils.data as data
import json
import os
from pathlib import Path

class WSIData(data.Dataset):
    def __init__(self, features_dir, stage, task, json_name, classes_to_use,
                 oversample_minority=False, file_manager=None):
        if json_name is None:
            json_name = f'\\{self.stage}.json'
        self.oversample_minority = oversample_minority
        self.features_dir = features_dir
        self.stage = stage
        self.num_classes = len(classes_to_use)
        self.classes_to_use = classes_to_use
        self.json_name = json_name
        self.all_slides = json.load(open(self.json_name))
        self.slides = []
        self.by_class = {}
        self.file_manager = file_manager

        for slide in self.all_slides:
            if slide['image_tag'] not in self.classes_to_use:
                continue

            base_name = os.path.basename(slide['image_path']).replace('.svs', '')

            base_name += '__' + slide['image_tag']
            
            to_search = os.path.join(self.features_dir, base_name + '.pt')
            if os.path.isfile(to_search):
                WSIData.add_to_dict(self.by_class, slide['dataset'], slide)
        self.create_slide_list()

    def get_counts_by_class(self):
        counts_by_class = {}
        for my_class in self.by_class.keys():
            counts_by_class[my_class] = len(self.by_class[my_class])
        return counts_by_class
    
    def create_slide_list(self):
        if self.oversample_minority:
            counts_by_class = self.get_counts_by_class()
            max_count = max(counts_by_class.values())
            print('Counts prior oversample', counts_by_class)
            for my_class in counts_by_class.keys():
                oversample_factor = int(max_count/counts_by_class[my_class])
                if oversample_factor > 1:
                    self.by_class[my_class] *= oversample_factor
                    random.shuffle(self.by_class[my_class])
            counts_by_class = self.get_counts_by_class()
            print('After oversample', counts_by_class)
        self.slides = []
        for my_class in self.by_class:
            self.slides += self.by_class[my_class]
        random.shuffle(self.slides)
        
    def add_to_dict(my_dict, my_class, data):
        if my_class in my_dict:
            my_dict[my_class].append(data)
        else:
            my_dict[my_class] = [data]
    
    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        slide_id = os.path.basename(self.slides[idx]['image_path']).replace('.svs', '')
        sulfix = self.slides[idx]['image_tag']
        slide_id += '__' + sulfix

        label = self.classes_to_use.index(self.slides[idx]['image_tag'])
    
        full_path = os.path.join(self.features_dir, f'{slide_id}.pt')
        if self.file_manager is not None:
            features = torch.load(self.file_manager.use_file(full_path))
        else:
            features = torch.load(full_path)
        participant = self.slides[idx]['participant']

        return features, label, participant