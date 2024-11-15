import sqlite3

con = sqlite3.connect("mimic4.db")
cur = con.cursor()
try:
    cur.execute(""" SELECT
  p.subject_id AS patient_id,
  p.gender AS geslacht,
  p.dod AS geboortedatum,
  a.hadm_id AS opname_nummer,
  a.admittime AS zkh_opn_start,
  a.insurance AS herkomst,
  a.marital_status,
  d.icd_code AS diagnose,
  d.icd_version,
  tr.intime AS start_operatie,
  tr.outtime,
  JULIANDAY(tr.outtime) - JULIANDAY(tr.intime) AS duur_ok
FROM
  patients p
JOIN
  admissions a
ON
  p.subject_id = a.subject_id
JOIN
  diagnoses_icd d
ON
  a.hadm_id = d.hadm_id
JOIN
  transfers tr
ON
  a.hadm_id = tr.hadm_id
""")
except sqlite3.Error as e:
    print(f"An error occurred: {e}")


# get all rows
rows = cur.fetchall()
#save the data in a csv file
with open('DESIRE.csv', 'w') as f:
    for row in rows:
        f.write(str(row) + '')

#close the connection
con.close()
