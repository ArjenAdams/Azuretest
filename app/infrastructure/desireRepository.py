from datetime import datetime, date

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import text

from backend.app.infrastructure.database import engine


class DesireRepository:
    def __init__(self):
        self.db = engine

    def close(self):
        self.db.close()

    def get_data_pd(self):


        #print("Connected to database successfully!")
        #print("Executing query...")

        query = text("""SELECT
  p.subject_id AS patient_id,
  p.gender AS geslacht,
  p.anchor_age + EXTRACT(YEAR FROM CAST(a.admittime AS TIMESTAMP)) - p.anchor_year AS leeftijd,
  a.admittime AS zkh_opn_start,
  a.hospital_expire_flag AS ziekenhuis_verval_vlag,
  a.dischtime AS zkh_ontheem_start,
  a.admission_type AS opname_type,
  a.discharge_location, 
  a.insurance AS verzekering,
  a.insurance AS herkomst,
  a.marital_status,
  d.icd_code AS diagnose_code,
  d.icd_version AS diagnose_versie,
  p.dod AS overlijdensdatum,
  tr.intime AS start_operatie,
  tr.outtime,


presc.drug AS medicijn,
EXTRACT(EPOCH FROM (tr.outtime::timestamp - tr.intime::timestamp)) AS duur_ok
FROM
  mimiciv_hosp.patients p
JOIN
  mimiciv_hosp.admissions a
ON
  p.subject_id = a.subject_id
JOIN
  mimiciv_hosp.diagnoses_icd d 
ON
  a.hadm_id = d.hadm_id
JOIN
  mimiciv_hosp.d_icd_diagnoses icd 
ON
  d.icd_code = icd.icd_code AND d.icd_version = icd.icd_version
JOIN
  mimiciv_hosp.transfers tr 
ON
  a.hadm_id = tr.hadm_id
JOIN
  (SELECT * FROM mimiciv_hosp.prescriptions LIMIT 1000) presc 
ON
  a.hadm_id = presc.hadm_id

ORDER BY p.subject_id DESC

LIMIT 1000;""")


        # Execute the SQL query
        with self.db.connect() as con:

            rs = con.execute(query) #Deze method print in de console...
            tuples = rs.fetchall()
            column_names = rs.keys()

        # Create a pandas DataFrame
        df = pd.DataFrame(tuples, columns=column_names)

        #print("Query executed successfully!")
        #print(df.head())
        return df

    def get_entities_by_searchvalues(self, searchValueList, limit=1000, filter_param=None, filter_value=None):
        query = """
                         SELECT
          p.subject_id AS patient_id,
          p.gender AS geslacht,
          p.anchor_age + EXTRACT(YEAR FROM CAST(a.admittime AS TIMESTAMP)) - p.anchor_year AS leeftijd,
          a.admittime AS zkh_opn_start,
          a.hospital_expire_flag AS ziekenhuis_verval_vlag,
          a.dischtime AS zkh_ontheem_start,
          a.admission_type AS opname_type,
          a.discharge_location, 
          a.insurance AS verzekering,
          a.insurance AS herkomst,
          a.marital_status,
          d.icd_code AS diagnose_code,
          d.icd_version AS diagnose_versie,
          p.dod AS overlijdensdatum,
          tr.intime AS start_operatie,
          tr.outtime,


        presc.drug AS medicijn,
        EXTRACT(EPOCH FROM (tr.outtime::timestamp - tr.intime::timestamp)) AS duur_ok
        FROM
          mimiciv_hosp.patients p
        JOIN
          mimiciv_hosp.admissions a
        ON
          p.subject_id = a.subject_id
        JOIN
          mimiciv_hosp.diagnoses_icd d 
        ON
          a.hadm_id = d.hadm_id
        JOIN
          mimiciv_hosp.d_icd_diagnoses icd 
        ON
          d.icd_code = icd.icd_code AND d.icd_version = icd.icd_version
        JOIN
          mimiciv_hosp.transfers tr 
        ON
          a.hadm_id = tr.hadm_id
        JOIN
          (SELECT * FROM mimiciv_hosp.prescriptions LIMIT 1000) presc 
        ON
          a.hadm_id = presc.hadm_id
        WHERE
            1=1
        """

        print(searchValueList.get('diagnose_code'))
        print(searchValueList)
        bind_params = {}
        if searchValueList.get('diagnose_code') != "":
            query += " AND d.icd_code = :diagnose_code"
            bind_params['diagnose_code'] = searchValueList.get('diagnose_code')

        if searchValueList.get('diagnose_versie') != '':
            query += " AND d.icd_version = :diagnose_versie"
            bind_params['diagnose_versie'] = searchValueList.get('diagnose_versie')


        if searchValueList.get('discharge_location') != '':
            query += " AND a.discharge_location = :discharge_location"
            bind_params['discharge_location'] = searchValueList.get('discharge_location')


        if searchValueList.get('duur_ok') != '':
            query += " AND EXTRACT(EPOCH FROM (tr.outtime::timestamp - tr.intime::timestamp)) = :duur_ok"
            bind_params['duur_ok'] = searchValueList.get('duur_ok')

        if searchValueList.get('geslacht') != '':
            query += " AND p.gender = :geslacht"
            bind_params['geslacht'] = searchValueList.get('geslacht')

        if searchValueList.get('herkomst') != '':
            query += " AND a.insurance = :herkomst"
            bind_params['herkomst'] = searchValueList.get('herkomst')

        if searchValueList.get('leeftijd') != '':
            query += " AND p.anchor_age + EXTRACT(YEAR FROM CAST(a.admittime AS TIMESTAMP)) - p.anchor_year = :leeftijd"
            bind_params['leeftijd'] = searchValueList.get('leeftijd')

        if searchValueList.get('marital_status') != '':
            query += " AND a.marital_status = :marital_status"
            bind_params['marital_status'] = searchValueList.get('marital_status')

        if searchValueList.get('medicijn') != '':
            query += " AND presc.drug = :medicijn"
            bind_params['medicijn'] = searchValueList.get('medicijn')

        if searchValueList.get('opname_type') != '':
            query += " AND a.admission_type = :opname_type"
            bind_params['opname_type'] = searchValueList.get('opname_type')

        if searchValueList.get('outtime') != '':
            query += " AND tr.outtime = :outtime"
            bind_params['outtime'] = searchValueList.get('outtime')

        if searchValueList.get('overlijdensdatum') != '':
            query += " AND p.dod = :overlijdensdatum"
            bind_params['overlijdensdatum'] = searchValueList.get('overlijdensdatum')

        if searchValueList.get('patient_id') != '':
            query += " AND p.subject_id = :patient_id"
            bind_params['patient_id'] = searchValueList.get('patient_id')

        if searchValueList.get('start_operatie') != '':
            query += " AND tr.intime = :start_operatie"
            bind_params['start_operatie'] = searchValueList.get('start_operatie')

        if searchValueList.get('verzekering') != '':
            query += " AND a.insurance = :verzekering"
            bind_params['verzekering'] = searchValueList.get('verzekering')

        if searchValueList.get('ziekenhuis_verval_vlag') != '':
            query += " AND a.hospital_expire_flag = :ziekenhuis_verval_vlag"
            bind_params['ziekenhuis_verval_vlag'] = searchValueList.get('ziekenhuis_verval_vlag')

        if searchValueList.get('zkh_ontheem_start') != '':
            query += " AND a.dischtime = :zkh_ontheem_start"
            bind_params['zkh_ontheem_start'] = searchValueList.get('zkh_ontheem_start')

        if searchValueList.get('zkh_opname_start') is not None:
            query += " AND a.admittime = :zkh_opname_start"
            bind_params['zkh_opname_start'] = searchValueList.get('zkh_opname_start')


        query += "\n ORDER BY p.subject_id DESC"


        if filter_param and filter_value:
            query += f" WHERE {filter_param} = :filter_value"
            bind_params['filter_value'] = filter_value

        query += " LIMIT :limit"
        bind_params['limit'] = limit

        with self.db.connect() as con:
            rs = con.execute(text(query).bindparams(**bind_params))

            results = [{column: value for column, value in row._mapping.items()} for row in rs]

            # Transform all the datetime and date objects into strings with 'yyyy-mm-dd hh-mm-ss' format (datetime and date can not be parsed into json)
            formatted_data = [
                {k: v.strftime('%Y-%m-%d %H:%M:%S') if isinstance(v, (datetime, date)) else v for k, v in entry.items()}
                for entry in results]

        return formatted_data

    def get_similarity_list(self, data, target_index,
                            method='jaccard',
                            sort='descending',
                            n_items=-1,
                            unique_only=False,
                            exclude_self=True
                            ):

        one_hot_data = OneHotEncoder(dtype=bool).fit_transform(data).toarray()

        similarity_df = data.copy()
        similarity_df['similarity'] = list(
            1 - pairwise_distances(one_hot_data, Y=np.array(one_hot_data[target_index]).reshape(1, -1), metric=method)[
                :,
                0])
        similarity_df['n_duplicates'] = data.groupby(list(data.columns), sort=False)[(list(data.columns))[0]].transform(
            'size') - 1
        similarity_df['index'] = similarity_df.index

        if unique_only:
            similarity_df = similarity_df.drop_duplicates(
                subset=list(similarity_df.drop(columns=['index', 'similarity', 'n_duplicates']).columns))

        if exclude_self:
            similarity_df = similarity_df.drop(index=similarity_df.iloc[target_index].name)

        similarity_list = similarity_df[['index', 'similarity', 'n_duplicates']].to_dict('records')

        if sort == 'descending':
            sort_direction = -1
        elif sort == 'ascending':
            sort_direction = 1
        else:
            return similarity_list[:n_items]

        return sorted(similarity_list, key=lambda d: d['similarity'])[::sort_direction][:n_items]

    # def get_data(self, limit=1000, filter_param=None, filter_value=None):


    # Function to preprocess new data
    def get_data(self, limit=1000, filter_param=None, filter_value=None):

        query = """
                 SELECT
  p.subject_id AS patient_id,
  p.gender AS geslacht,
  p.anchor_age + EXTRACT(YEAR FROM CAST(a.admittime AS TIMESTAMP)) - p.anchor_year AS leeftijd,
  a.admittime AS zkh_opn_start,
  a.hospital_expire_flag AS ziekenhuis_verval_vlag,
  a.dischtime AS zkh_ontheem_start,
  a.admission_type AS opname_type,
  a.discharge_location, 
  a.insurance AS verzekering,
  a.insurance AS herkomst,
  a.marital_status,
  d.icd_code AS diagnose_code,
  d.icd_version AS diagnose_versie,
  p.dod AS overlijdensdatum,
  tr.intime AS start_operatie,
  tr.outtime,


presc.drug AS medicijn,
EXTRACT(EPOCH FROM (tr.outtime::timestamp - tr.intime::timestamp)) AS duur_ok
FROM
  mimiciv_hosp.patients p
JOIN
  mimiciv_hosp.admissions a
ON
  p.subject_id = a.subject_id
JOIN
  mimiciv_hosp.diagnoses_icd d 
ON
  a.hadm_id = d.hadm_id
JOIN
  mimiciv_hosp.d_icd_diagnoses icd 
ON
  d.icd_code = icd.icd_code AND d.icd_version = icd.icd_version
JOIN
  mimiciv_hosp.transfers tr 
ON
  a.hadm_id = tr.hadm_id
JOIN
  (SELECT * FROM mimiciv_hosp.prescriptions LIMIT 1000) presc 
ON
  a.hadm_id = presc.hadm_id

ORDER BY p.subject_id DESC"""

        bind_params = {}

        if filter_param and filter_value:
            query += f" WHERE {filter_param} = :filter_value"
            bind_params['filter_value'] = filter_value

        query += " LIMIT :limit"
        bind_params['limit'] = limit

        with self.db.connect() as con:
            rs = con.execute(text(query).bindparams(**bind_params))

            results = [{column: value for column, value in row._mapping.items()} for row in rs]

            # Transform all the datetime and date objects into strings with 'yyyy-mm-dd hh-mm-ss' format (datetime and date can not be parsed into json)
            formatted_data = [{k: v.strftime('%Y-%m-%d %H:%M:%S') if isinstance(v, (datetime, date)) else v for k, v in entry.items()} for entry in results]

        return formatted_data

