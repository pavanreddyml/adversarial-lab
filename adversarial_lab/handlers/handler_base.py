from abc import ABC, abstractmethod

from typing import Dict

class HandlerBase(ABC):
    def __init__(self, 
                 batch_size, 
                 *args, 
                 **kwargs):
        self.batch_size = batch_size

    @abstractmethod
    def load(self, 
             *args, 
             **kwargs):
        pass
    
    @abstractmethod
    def write(self, 
              data, 
              *args, 
              **kwargs):
        pass

    @abstractmethod
    def get_same_read_write(self):
        pass

    @abstractmethod
    def get_class_names(self):
        pass

    @abstractmethod
    def get_num_entries_by_class(self, class_name: str) -> Dict[str, int]:
        pass

    @abstractmethod
    def get_total_samples(self):
        pass

    def get_batch(self, class_name: str = None):
        data = []
        for _ in range(self.batch_size):
            sample = self.load(class_name=class_name)
            if sample is not None:
                data.append(sample)
            else:
                break
            
        if len(data) == 0:
            return None
        return data
    
    def write_batch(self, data):
        for item in data:
            self.write(*item)