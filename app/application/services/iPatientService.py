from abc import ABC, abstractmethod

from backend.app.domain.entities.patient import Patient


class IPatientService(ABC):
    @abstractmethod
    def get_patient(self, name: str) -> Patient:
        pass

