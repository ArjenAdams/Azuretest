from ..domain.entities import Prediction
from ..infrastructure.database import SessionLocal


#TODO: save the prediction in the database with the input features and the output prediction
class IPredictionRepository:
    def __init__(self):
        self.db = SessionLocal()

    def close(self):
        self.db.close()

    def add_entity(self, entity: Prediction):
        self.db.add(entity)
        self.db.flush()  # to get the id
        self.db.commit()
        return entity.id

    def get_entity(self, id: int) -> Prediction:
        return self.db.query(Prediction).filter(Prediction.id == id).first()

    def get_all_entities(self):
        return self.db.query(Prediction).all()

