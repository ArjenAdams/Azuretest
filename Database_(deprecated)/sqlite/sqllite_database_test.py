import sqlite3



con = sqlite3.connect("mimic4.db")
cur = con.cursor()
# reading all table names
table_list = [a for a in cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
# here is you table list
print(table_list)

# get d_items table and save it as a csv file
cur.execute("SELECT * FROM admissions")
rows = cur.fetchall()
with open('admissions.csv', 'w') as f:
    for row in rows:
        f.write(str(row) + '')


