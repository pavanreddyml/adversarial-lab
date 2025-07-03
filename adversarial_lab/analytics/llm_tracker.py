from . import Tracker

import warnings
import numpy as np

from typing import Dict, List, Literal, Any

class LLMTracker(Tracker):
    _columns = {
        "epoch_llm_prompt": "str",
        "epoch_llm_prompt_by_batch": "json",
        "epoch_llm_noise": "json",
        "epoch_llm_noise_by_batch": "json",
        "epoch_llm_response": "str",
        "epoch_llm_response_by_batch": "json"
    }

    def __init__(self,
                 track_batch: bool = True,
                 track_epoch: bool = True,
                 ) -> None:
        super().__init__(track_batch=track_batch, track_epoch=track_epoch)

    def post_batch(self,
                   *args,
                   **kwargs
                   ) -> None:
        prompts = kwargs.get("original_sample", [])
        noise = kwargs.get("noise", [])
        predictions = kwargs.get("predictions", [])

        if not self.track_batch:
            return

        if len(self.epoch_llm_prompt_by_batch) == 0:
            epoch_val = -1
        else:
            epoch_val = max(self.epoch_llm_prompt_by_batch.keys()) + 1
            
        self.epoch_llm_prompt_by_batch[epoch_val] = prompts
        self.epoch_llm_noise_by_batch[epoch_val] = noise
        self.epoch_llm_response_by_batch[epoch_val] = predictions

    def post_epoch(self,
                   *args,
                   **kwargs
                   ) -> None:
        prompt = kwargs.get("original_sample", "")
        noise = kwargs.get("noise", {})
        prediction = kwargs.get("predictions", "")

        if not self.track_epoch:
            return  
        
        self.epoch_llm_prompt = prompt
        self.epoch_llm_noise = noise
        self.epoch_llm_response = prediction

        
    def serialize(self) -> Dict:
        data = {}


        if self.track_batch:
            data["epoch_llm_prompt_by_batch"] = self.epoch_llm_prompt_by_batch
            data["epoch_llm_response_by_batch"] = self.epoch_llm_response_by_batch

        if self.track_epoch:
            data["epoch_llm_prompt"] = self.epoch_llm_prompt
            data["epoch_llm_response"] = self.epoch_llm_response

        return data
    
    def reset_values(self) -> None:
        self.epoch_llm_prompt = ""
        self.epoch_llm_prompt_by_batch: Dict[int, List[str]] = {}
        self.epoch_llm_noise: Dict[str, Any] = {}
        self.epoch_llm_noise_by_batch: Dict[int, List[Dict]] = {}
        self.epoch_llm_response = ""
        self.epoch_llm_response_by_batch: Dict[int, List[str]] = {}

