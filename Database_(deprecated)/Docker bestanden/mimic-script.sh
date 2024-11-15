#!/bin/bash
createdb mimiciv
psql -d mimiciv -f /tmp/mimic-iv/buildmimic/postgres/create.sql
psql -d mimiciv -v ON_ERROR_STOP=1 -v mimic_data_dir=/tmp -f mimic-iv/buildmimic/postgres/load_gz.sql
psql -d mimiciv -v ON_ERROR_STOP=1 -v mimic_data_dir=/tmp -f mimic-iv/buildmimic/postgres/constraint.sql
psql -d mimiciv -v ON_ERROR_STOP=1 -v mimic_data_dir=/tmp -f mimic-iv/buildmimic/postgres/index.sql
