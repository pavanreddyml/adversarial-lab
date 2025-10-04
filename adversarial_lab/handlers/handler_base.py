from abc import ABC, abstractmethod


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

    def get_batch(self):
        data = []
        for _ in range(self.batch_size):
            sample = self.load()
            if sample is not None:
                data.append(sample)
            else:
                break
            
        if len(data) == 0:
            return None
        return data