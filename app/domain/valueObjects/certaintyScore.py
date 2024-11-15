class CertaintyScore:
    def __init__(self, value):
        if 0 <= value <= 100:
            self.value = value
        else:
            raise ValueError("Certainty score must be between 0 and 100")

    def to_integer(self):
        return int(self.value)
