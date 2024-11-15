from backend.app.infrastructure.desireRepository import DesireRepository


class DesireService():
    def __init__(self):
        self.desire_repository = DesireRepository()

    def get_dataset(self):
        return self.desire_repository.get_data()
