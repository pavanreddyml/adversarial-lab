from typing import Dict
from abc import ABC, abstractmethod


class Tracker(ABC):
    @property
    def _columns(self) -> Dict:
        return {}

    def __init__(self,
                 track_batch: bool = True,
                 track_epoch: bool = True,) -> None:
        """
        Module for tracking, storing and serializeing data during the attack process.

        Args:
            track_batch (bool, optional): If True, the tracker will track data after each batch. Defaults to True.
            track_epoch (bool, optional): If True, the tracker will track data after each epoch. Defaults to True.
        """

        if not hasattr(self, '_columns') or not isinstance(self._columns, dict):
            raise AttributeError(
                "The '_columns' attribute must be defined as a dictionary in the subclass.")

        self.track_batch = track_batch
        self.track_epoch = track_epoch

        self.reset_values()

    def pre_attack(self,
                   *args,
                   **kwargs
                   ) -> None:
        """
        Updates the values of tracker before starting the attack.

        Keyword Args:
            param1 (type, optional): Description of param1.
            param2 (type, optional): Description of param2.
            param3 (type, optional): Description of param3.
        """
        pass

    def post_batch(self,
                   batch_num: int,
                   *args,
                   **kwargs
                   ) -> None:
        """
        Updates the values of tracker after a batch. 
        
        Args:
            batch_num (int): The current batch number.

        Keyword Args:
            param1 (type, optional): Description of param1.
            param2 (type, optional): Description of param2.
            param3 (type, optional): Description of param3.
        """
        pass

    def post_epoch(self,
                   *args,
                   **kwargs
                   ) -> None:
        """
        Updates the values of tracker after an epoch.
        
        Keyword Args:
            param1 (type, optional): Description of param1.
            param2 (type, optional): Description of param2.
            param3 (type, optional): Description of param3.
        """
        pass

    def post_attack(self,
                    *args,
                    **kwargs
                    ) -> None:
        """
        Updates the values of tracker after the attack.

        Keyword Args:
            param1 (type, optional): Description of param1.
            param2 (type, optional): Description of param2.
            param3 (type, optional): Description of param3.
        """
        pass

    @abstractmethod
    def serialize(self) -> Dict:
        """
        Serializes the tracked data into a dictionary.

        Returns:
            Dict: A dictionary containing the tracked data.
        """
        pass

    @abstractmethod
    def reset_values(self) -> None:
        """
        Resets the tracked values to their initial state.
        """
        pass
