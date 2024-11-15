from abc import ABC, abstractmethod

from backend.app.domain.entities import HumanBodyFeatures as humanB


class IPredictionService(ABC):
    @abstractmethod
    def generate_prediction(self, features: humanB) -> humanB:
        pass

    @abstractmethod
    def get_predictions(self) -> humanB:
        pass
