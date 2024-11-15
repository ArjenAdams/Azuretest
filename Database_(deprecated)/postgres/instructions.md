# DEPRECATED - DO NOT USE 2024-05-24

# Local PostgreSQL Database Setup Instructions (WINDOWS)

## Installing PostgreSQL
Firstly, make sure you have [PostgreSQL installed](https://www.postgresql.org/download/). For your
convienience, when the installer asks for a password set it to 'postgres'. Keep note of where you install
postgres (this will be needed for your environment variables in the next step).

## Setting up environment variables
You have to edit your system environment variables in order to use psql (a postgres CLI)

- Search for 'system environment' in your search bar

![System Enviroment Variables in search bar](../../Resources/readme_images/sys_env_search.png)
- Click on 'Environment Variables'

![Environment Variabeles](../../Resources/readme_images/env_var.png)
- Under System Variables, select Path and go to 'Edit...'

![Edit Path](../../Resources/readme_images/edit_path.png)
- Click on 'New' and add 2 seperate paths.
  - C:\Program Files\PostgreSQL\16\bin (Or a different path depending on your Postgres installation location)
  - C:\Program Files\PostgreSQL\16\lib (Or a different path depending on your Postgres installation location)

  ![img.png](../../Resources/readme_images/path_vars.png)
- Hit 'OK' until all three screens are closed

## Preparing the database
First, we will create the database itself:
1. Go to your search bar and search for 'psql' and open the shell. 
2. Hit enter on all fields but password (this only works if you kept the standard postgres installation options)
3. login to your postgres user with the password you used when installing PostgreSQL (Should be 'postgres'). Your shell should look like this at this point:

![psql Login](../../Resources/readme_images/psql_login.png)
4. Run the following command: 

``DROP DATABASE IF EXISTS mimic4;CREATE DATABASE mimic4 OWNER postgres;``

This will create a database named mimic4 on your postgres server. Your shell should look like this now:

![Create Database](../../Resources/readme_images/create_db.png)

Secondly, we will create the schemas and tables for the database 'mimic4'.
1. In your terminal, navigate to ``Dashboard-XAI-trust\Database\postgres`` and enter the following command: 

``psql -U <YOUR USERNAME> -d mimic4 -f create.sql``

Here your username should be 'postgres'. The terminal should ask you for your password, which also should be 'postgres'. Your terminal should look like this:
![Creating the schemas and tables](../../Resources/readme_images/create_schemas_tables.png)

2. Download the prescriptions.csv.gz file from this [Google Drive](https://drive.google.com/drive/folders/1-4xmOEQiX0hZPv56rPPl83hTcNBLUjBG?usp=sharing) folder. Then add the file to the ``Database/csv_data/hosp`` directory
   (This file was too large to add to the GitHub repository)
3. Extract all the files in the Database/csv_data directory into their corresponding folders (they should now be .csv instead of .csv.gz):

![Exract Zips](../../Resources/readme_images/extract_zips.png)

4. While in the same terminal location (``Dashboard-XAI-trust\Database\postgres``), enter the following command:

``psql -U <YOUR USERNAME> -d mimic4 -v ON_ERROR_STOP=1 -v mimic_data_dir=../csv_data -f load.sql``
    
This will load the .csv files in ``Database/csv_data`` into the database tables. Loading in the data could take a couple of minutes!
