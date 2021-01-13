#!/usr/bin/python3.8
# Anmol Kapoor
'''
Intermediary Part 1 in this project is a tool to create a database that
combines all of the other databases in the 'jobs' folder.
'''

import sqlite3 as sql
import sys
from os import chdir, listdir, path, remove

def create_connection(db_file):
    conn = None
    try:
        conn = sql.connect(db_file)
        cur = conn.cursor()
    except sql.Error as e:
        print(e)
        if path.exists('jobs.db'):
            main_curs.close()
            main_cnxn.close()
            remove('jobs.db')
        sys.exit(1)
    return conn, cur

main_cnxn, main_curs = create_connection('jobs.db')
main_curs.execute("select name from sqlite_master where type='table' and name = 'jobs'")
fetcher = main_curs.fetchone()
print(fetcher)
if fetcher and 'jobs' in fetcher:
    main_curs.execute('drop table jobs')
    main_cnxn.commit()
main_curs.execute('create table jobs (company nvarchar(20), title nvarchar(120), link nvarchar(175))')
main_cnxn.commit()

insertion_cmd = "insert into jobs(company, title, link) values "
notFirst = False
for database_file in listdir('jobs/'):
    next_cnxn, next_curs = create_connection('jobs/' + database_file)
    next_curs.execute('select company, title, link from co_jobs')
    job_opportunities = next_curs.fetchall()
    for (company, title, link) in job_opportunities:
        if notFirst:
            insertion_cmd += ','
        insertion_cmd += f"('{company}', '{title}', '{link}') "
        notFirst = True
    print("Adding data from", database_file)
main_curs.execute(insertion_cmd)
main_cnxn.commit()
