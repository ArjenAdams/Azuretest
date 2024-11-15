# Dashboard Setup

## Benodigde software om te installeren
- Python 3.9, 3.10 of 3.11
- Node.js (Laatste stabiele versie)
- NPM (Node Package Manager)
- IDE (PyCharm wordt aanbevolen)

## Stappen om het dashboard te starten
1. Clone de repository naar je lokale machine. [Instructies](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository#cloning-a-repository)
2. Open de repository in je IDE.
3. Maak een virtuele omgeving in de root van de repository met behulp van het requirements.txt-bestand. [Instructies](https://docs.python.org/3/library/venv.html)
4. Als je PyCharm gebruikt, volg dan deze instructies.
   1. Voeg de Python-interpreter toe in PyCharm.
   2. Synchroniseer het requirements.txt-bestand dat zich in de setup_instructions bevindt.
   3. ![img.png](img.png)
   4. Als deze optie om een of andere reden niet werkt, voer dan dit commando uit **pip install -r ./Setup_instructions/requirements.txt**.
5. Installeer de Node.js-pakketten in de frontend-map met behulp van het package.json-bestand met het commando: **npm install --force** [Verdere instructies](https://docs.npmjs.com/cli/v7/commands/npm-install)

# Het dashboard starten
1. Voer het bestand `run.py` uit in de map met het label “backend” (d.w.z. (<Projectroot>\Dashboard-XAI-trust\backend\run.py).
2. Open een nieuwe terminal in de IDE.
3. Voer in de nieuwe terminal `cd ./frontend` in.
4. Voer `npm start` in.

Het dashboard zou nu moeten draaien en kan worden geopend via de volgende URL: http://localhost:3000/.

# Het verkrijgen van de SAD-data uit MIMIC-IV
1. De data die nodig is voor het SAD-model staat in deze [Github-repo](https://github.com/bbycat927/SAD). Download en unzip de ZIP van de repo en gebruik pandas met het commando `pd.read_strata('path_to_MIMIC-IV.dta')` om de data in een pandas dataframe te laden.

# Waarom MIMIC-IV?

MIMIC-IV is een grote dataset die meer dan 2 miljoen records bevat van patiënten die zijn opgenomen op de IC in de Verenigde Staten. De dataset bevat een breed scala aan informatie over de patiënten, zoals demografie, vitale functies, laboratoriumresultaten en meer.

Er is veel onderzoek gedaan naar de MIMIC-IV-dataset, hierdoor kun je veel artikelen en papers vinden die je helpen de dataset te begrijpen en te gebruiken. Wij hebben de paper genaamd "**Development of a machine learning-based prediction model for sepsis-associated delirium in the intensive care unit**" gebruikt, waarin ze de MIMIC-IV-dataset gebruikten om sepsis-geassocieerd delirium (SAD) op de IC te voorspellen. We hebben dezelfde dataset gebruikt om het model opnieuw te creëren en de resultaten in het dashboard te visualiseren.

De dataset is gratis te gebruiken, maar vereist een account en het volgen van een kleine cursus op de website om toegang te krijgen en de data te downloaden. Zie de volgende stappen om toegang te krijgen tot de data.

# Toegang krijgen tot MIMIC-IV
1. Ga naar [PhysioNet.org](https://physionet.org/). Je moet een account aanmaken en je wordt gevraagd om een formulier in te vullen met je persoonlijke informatie, een korte beschrijving van je beoogde gebruik van de data en het e-mailadres van de producteigenaar, meestal die van Danielle (d.sent@tue.nl). Het kan een paar dagen duren voordat je account is goedgekeurd.

Hier is een deel van het formulier dat je moet invullen:
![alt text](image.png)

2. Nadat je toegang hebt verkregen, moet je deze cursus voltooien [CITI Course Tutorial](https://physionet.org/about/citi-course/). Nadat je de cursus hebt voltooid en de certificering aan je account hebt toegevoegd, kun je de data downloaden.

3. Download de MIMIC-IV-data van de [PhysioNet-website](https://physionet.org/content/mimiciv/2.2/). Je hebt 7zip nodig om de .gz-bestanden uit te pakken.

4. Pak de bestanden uit in de map `Database/csv_data`. De bestanden moeten in .csv-formaat zijn. Als de map niet bestaat, maak deze dan aan in de `Database`-directory.

5. Je kunt de data nu gebruiken om verder te gaan met het project.
