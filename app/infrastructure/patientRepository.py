from database import SessionLocal
from ..domain.entities.patient import Patient


class IPatientRepository:
    def __init__(self):
        self.db = SessionLocal()

    def close(self):
        self.db.close()

    def add_entity(self, entity: Patient):
        self.db.add(entity)
        self.db.flush()  # flush() is used to generate the id
        self.db.commit()
        return entity

    def update_entity(self, entity: Patient):
        self.db.commit()
        return entity

    def get_entity(self, id: int) -> Patient:
        return self.db.query(Patient).filter(Patient.id == id).first()

    def get_entities_by_searchvalues(self) -> Patient:
        return self.db.query(Patient).filter(Patient.id == 10031687).first()
