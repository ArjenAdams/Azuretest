from __future__ import annotations

import datetime

from sqlalchemy import Column, Integer, String
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

from ...infrastructure.database import Base as base


class Prediction(base):
    __tablename__ = "prediction"
    id = Column(Integer, primary_key=True)
    human_body_features = relationship("HumanBodyFeatures", back_populates="predictions", lazy='joined')
    human_body_features_id = Column(Integer, ForeignKey('human_body_features.id'))
    certainty_score = Column(Integer)
    medical_sickness = Column(String)
    medical_sickness_description = Column(String)
    created_at = Column(String)

    def __init__(self, human_body_features, certainty_score, medical_sickness, medical_description):
        self.human_body_features = human_body_features
        self.certainty_score = certainty_score
        self.medical_sickness = medical_sickness
        self.medical_sickness_description = medical_description
        self.created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


