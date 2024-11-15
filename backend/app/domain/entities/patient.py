import datetime

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from ...infrastructure.database import Base as base


class Patient(base):
    __tablename__ = "patient"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    birthtime = Column(String)
    patientId = Column(String)
    features = relationship("HumanBodyFeatures", backref="patient")
    created_at = Column(String)

    def __init__(self, name: str, birth_date: datetime, patient_id: str, features: list):
        self.name = name
        self.birthtime = birth_date
        self.patientId = patient_id
        self.features = features
        self.created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def patient_to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'features': [feature.features_to_dict() for feature in self.features],
            'created_at': self.created_at
        }
