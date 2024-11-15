from __future__ import annotations

import datetime

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from ...infrastructure.database import Base


class HumanBodyFeatures(Base):
    __tablename__ = "human_body_features"
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patient.id'))  # New field
    predictions = relationship("Prediction", back_populates="human_body_features", cascade="all, delete-orphan")
    body_mass_index = Column(String)
    asa_score = Column(Integer)
    respiratory_rate = Column(Integer)
    oxygen_saturation = Column(Integer)
    heart_rate = Column(Integer)
    blood_pressure = Column(String)
    body_temperature = Column(Integer)
    pain_level = Column(Integer)
    metabolic_rate = Column(Integer)
    physical_activity = Column(Integer)
    created_at = Column(String)

    def __init__(self, patient_id: int, body_mass_index: str, asa_score: int,
                 respiratory_rate: int, oxygen_saturation: int, heart_rate: int,
                 blood_pressure: str, body_temperature: int, pain_level: int,
                 metabolic_rate: int, physical_activity: int):
        self.patient_id = patient_id
        self.body_mass_index = body_mass_index
        self.asa_score = asa_score
        self.respiratory_rate = respiratory_rate
        self.oxygen_saturation = oxygen_saturation
        self.heart_rate = heart_rate
        self.blood_pressure = blood_pressure
        self.body_temperature = body_temperature
        self.pain_level = pain_level
        self.metabolic_rate = metabolic_rate
        self.physical_activity = physical_activity
        self.created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def features_to_dict(self):
        return {
            'body_mass_index': self.body_mass_index,
            'asa_score': self.asa_score,
            'respiratory_rate': self.respiratory_rate,
            'oxygen_saturation': self.oxygen_saturation,
            'heart_rate': self.heart_rate,
            'blood_pressure': self.blood_pressure,
            'body_temperature': self.body_temperature,
            'pain_level': self.pain_level,
            'metabolic_rate': self.metabolic_rate,
            'physical_activity': self.physical_activity
        }

