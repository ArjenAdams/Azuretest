import json

from pathlib import Path


class Config:
    pass


def getDatabaseURI():
    mod_path = Path(__file__).parent
    with open((mod_path / 'database.json').resolve()) as f:
        config = json.load(f)

    database_parameters = config['postgresql']
    URI = f"postgresql://{database_parameters['username']}:{database_parameters['password']}@{database_parameters['host']}:{database_parameters['port']}/{database_parameters['database']}"
    return URI
