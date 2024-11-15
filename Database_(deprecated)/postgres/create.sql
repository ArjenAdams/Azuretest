-------------------------------------------
-- Create the tables and MIMIC-IV schema --
-------------------------------------------

----------------------
-- Creating schemas --
----------------------

DROP SCHEMA IF EXISTS mimiciv_hosp CASCADE;
CREATE SCHEMA mimiciv_hosp;
DROP SCHEMA IF EXISTS mimiciv_icu CASCADE;
CREATE SCHEMA mimiciv_icu;
DROP SCHEMA IF EXISTS mimiciv_derived CASCADE;
CREATE SCHEMA mimiciv_derived;

---------------------
-- Creating tables --
---------------------

-- hosp schema

DROP TABLE IF EXISTS mimiciv_hosp.admissions;
CREATE TABLE mimiciv_hosp.admissions
(
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  admittime TIMESTAMP NOT NULL,
  dischtime TIMESTAMP,
  deathtime TIMESTAMP,
  admission_type VARCHAR(40) NOT NULL,
  admit_provider_id VARCHAR(10),
  admission_location VARCHAR(60),
  discharge_location VARCHAR(60),
  insurance VARCHAR(255),
  language VARCHAR(10),
  marital_status VARCHAR(30),
  race VARCHAR(80),
  edregtime TIMESTAMP,
  edouttime TIMESTAMP,
  hospital_expire_flag SMALLINT
);

DROP TABLE IF EXISTS mimiciv_hosp.diagnoses_icd;
CREATE TABLE mimiciv_hosp.diagnoses_icd
(
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  seq_num INTEGER NOT NULL,
  icd_code CHAR(7),
  icd_version SMALLINT
);

DROP TABLE IF EXISTS mimiciv_hosp.d_icd_diagnoses;
CREATE TABLE mimiciv_hosp.d_icd_diagnoses
(
  icd_code CHAR(7) NOT NULL,
  icd_version SMALLINT NOT NULL,
  long_title VARCHAR(255)
);

DROP TABLE IF EXISTS mimiciv_hosp.d_icd_procedures;
CREATE TABLE mimiciv_hosp.d_icd_procedures
(
  icd_code CHAR(7) NOT NULL,
  icd_version SMALLINT NOT NULL,
  long_title VARCHAR(222)
);

DROP TABLE IF EXISTS mimiciv_hosp.d_labitems;
CREATE TABLE mimiciv_hosp.d_labitems
(
  itemid INTEGER NOT NULL,
  label VARCHAR(50),
  fluid VARCHAR(50),
  category VARCHAR(50)
);

DROP TABLE IF EXISTS mimiciv_hosp.omr;
CREATE TABLE mimiciv_hosp.omr
(
  subject_id INTEGER NOT NULL,
  chartdate DATE NOT NULL,
  seq_num INTEGER NOT NULL,
  result_name VARCHAR(100) NOT NULL,
  result_value TEXT NOT NULL
);

DROP TABLE IF EXISTS mimiciv_hosp.patients;
CREATE TABLE mimiciv_hosp.patients
(
  subject_id INTEGER NOT NULL,
  gender CHAR(1) NOT NULL,
  anchor_age SMALLINT,
  anchor_year SMALLINT NOT NULL,
  anchor_year_group VARCHAR(20) NOT NULL,
  dod DATE
);

DROP TABLE IF EXISTS mimiciv_hosp.prescriptions;
CREATE TABLE mimiciv_hosp.prescriptions
(
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  pharmacy_id INTEGER NOT NULL,
  poe_id VARCHAR(25),
  poe_seq INTEGER,
  order_provider_id VARCHAR(10),
  starttime TIMESTAMP(3),
  stoptime TIMESTAMP(3),
  drug_type VARCHAR(20) NOT NULL,
  drug VARCHAR(255) NOT NULL,
  formulary_drug_cd VARCHAR(50),
  gsn VARCHAR(255),
  ndc VARCHAR(25),
  prod_strength VARCHAR(255),
  form_rx VARCHAR(25),
  dose_val_rx VARCHAR(100),
  dose_unit_rx VARCHAR(50),
  form_val_disp VARCHAR(50),
  form_unit_disp VARCHAR(50),
  doses_per_24_hrs REAL,
  route VARCHAR(50)
);

DROP TABLE IF EXISTS mimiciv_hosp.transfers;
CREATE TABLE mimiciv_hosp.transfers
(
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER,
  transfer_id INTEGER NOT NULL,
  eventtype VARCHAR(10),
  careunit VARCHAR(255),
  intime TIMESTAMP,
  outtime TIMESTAMP
);

-- icu schema

DROP TABLE IF EXISTS mimiciv_icu.d_items;
CREATE TABLE mimiciv_icu.d_items
(
  itemid INTEGER NOT NULL,
  label VARCHAR(100) NOT NULL,
  abbreviation VARCHAR(50) NOT NULL,
  linksto VARCHAR(30) NOT NULL,
  category VARCHAR(50) NOT NULL,
  unitname VARCHAR(50),
  param_type VARCHAR(20) NOT NULL,
  lownormalvalue FLOAT,
  highnormalvalue FLOAT
);

DROP TABLE IF EXISTS mimiciv_icu.icustays;
CREATE TABLE mimiciv_icu.icustays
(
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  stay_id INTEGER NOT NULL,
  first_careunit VARCHAR(255),
  last_careunit VARCHAR(255),
  intime TIMESTAMP,
  outtime TIMESTAMP,
  los FLOAT
);
