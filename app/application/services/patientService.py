from backend.app.application.services.iPatientService import IPatientService
from backend.app.domain.entities.patient import Patient
from backend.app.infrastructure.patientRepository import IPatientRepository
class PatientService(IPatientService):
    def __init__(self):
        self.patient_repository = IPatientRepository()

    def get_patient(self, patient_id: int) -> Patient:
        return self.patient_repository.get_entity(id=patient_id)

    def add_patient(self, patient: Patient) -> Patient:
        patient = self.patient_repository.add_entity(patient)
        return patient

    def get_patients_by_searchvalues(self) -> Patient:
        patients = self.patient_repository.get_entities_by_searchvalues()
        return patients


