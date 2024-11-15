import os
import psycopg2
from glob import glob
import pandas as pd
from sqlalchemy import create_engine
import urllib.parse

POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = r'b#!TyE4cP5^r@hM@Fs28'
encoded = urllib.parse.quote_plus(POSTGRES_PASSWORD)

POSTGRES_HOST = "31.220.74.113"
POSTGRES_PORT = "5432"
DATABASE_NAME = "mimic4"

THRESHOLD_SIZE = 5 * 10**7
CHUNKSIZE = 10**5

CONNECTION_STRING  = f"postgresql+psycopg2://{POSTGRES_USER}:{encoded}@{POSTGRES_HOST}:{POSTGRES_PORT}/{DATABASE_NAME}?connect_timeout=120"

engine = create_engine(CONNECTION_STRING)
engine.execute("DROP SCHEMA IF EXISTS mimic4 CASCADE;")

files = glob("**/*.csv*", recursive=True)

# Function to load data from a CSV file
def load_csv(file):
    print("Starting processing {}".format(file))
    folder, filename = os.path.split(file)
    tablename = filename.lower()
    if tablename.endswith('.gz'):
        tablename = tablename[:-3]
    if tablename.endswith('.csv'):
        tablename = tablename[:-4]
    try:
        if os.path.getsize(file) < THRESHOLD_SIZE:
            df = pd.read_csv(file)
            df.to_sql(tablename, engine, if_exists="replace", index=False)
        else:
            if not engine.has_table(tablename):
                for chunk in pd.read_csv(file, chunksize=CHUNKSIZE, low_memory=False):
                    chunk.to_sql(tablename, engine, if_exists="append", index=False)
        print("Finished processing {}".format(file))
    except Exception as e:
        print("Error while processing file: ", file)
        print(e)

# Process files sequentially
for file in files:
    load_csv(file)

# Connect to the PostgreSQL database using psycopg2
conn = psycopg2.connect(
    dbname=DATABASE_NAME,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
)

# Create a cursor object
cursor = conn.cursor()

# Define foreign key relationships between tables
relationships = [
    ("admissions", "subject_id", "patients", "subject_id"),
    ("admissions", "hadm_id", "icustays", "hadm_id"),
    ("admissions", "hadm_id", "transfers", "hadm_id"),
    ("icustays", "subject_id", "patients", "subject_id"),
    ("icustays", "icustay_id", "transfers", "icustay_id"),
]

# Drop foreign key constraints if they exist, and then re-add them
for relationship in relationships:
    source_table, source_column, target_table, target_column = relationship
    constraint_name = f"fk_{source_table}_{target_table}"
    try:
        cursor.execute(f"""
            ALTER TABLE {source_table}
            ADD CONSTRAINT {constraint_name}
            FOREIGN KEY ({source_column})
            REFERENCES {target_table} ({target_column});
        """)
        print(f"Foreign key constraint added between {source_table} and {target_table}")
    except Exception as e:
        print(f"Could not add foreign key constraint between {source_table} and {target_table}. Error: {e}")
    conn.commit()

cursor.close()
conn.close()





print("All done!")
