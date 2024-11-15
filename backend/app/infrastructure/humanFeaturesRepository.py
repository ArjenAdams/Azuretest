from backend.app.domain.entities import HumanBodyFeatures
from database import SessionLocal


class IPredictionRepository:
    def __init__(self):
        self.db = SessionLocal()

    def close(self):
        self.db.close()

    def add_entity(self, entity: HumanBodyFeatures):
        self.db.add(entity)
        self.db.flush()  # to get the id
        self.db.commit()
        return entity.id

    def get_entity(self, id: int) -> HumanBodyFeatures:
        return self.db.query(HumanBodyFeatures).filter(HumanBodyFeatures.id == id).first()
