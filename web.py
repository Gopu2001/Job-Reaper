#!/usr/bin/python3.8
# Anmol Kapoor

from flask import Flask, render_template, request
from flask_restful import Resource, Api
from os import getcwd
import sqlite3 as sql
import jinja2

app = Flask(__name__)
api = Api(app)

my_loader = jinja2.ChoiceLoader([app.jinja_loader, jinja2.FileSystemLoader(getcwd()),])
app.jinja_loader = my_loader

# create the json
class Jobbing(Resource):
    def get(self):
        connection = sql.connect('ml_jobs.db')
        cursor = connection.cursor()
        command = "select company, title, link from jobs"
        cursor.execute(command)
        opportunities = cursor.fetchall()
        cursor.close()
        connection.close()
        jobbing = {}
        for job in opportunities:
            jobber = {}
            jobber['company'] = job[0]
            jobber['title'] = job[1]
            jobber['application link'] = job[2]
            jobbing[opportunities.index(job)] = jobber
        return jobbing

api.add_resource(Jobbing, '/jobs.json')

# home page
@app.route('/')
def home_page(data=None):
    connection = sql.connect('ml_jobs.db')
    cursor = connection.cursor()
    command = "select distinct company from jobs"
    cursor.execute(command)
    data = cursor.fetchall()
    cursor.close()
    connection.close()
    return render_template('home.html', data=data)

# company page
@app.route('/companies/<company>/')
def sub_pages(company, data=None):
    base = '/companies/'
    connection = sql.connect('ml_jobs.db')
    cursor = connection.cursor()
    command = f"select title, link from jobs where company = '{company}'"
    cursor.execute(command)
    data = cursor.fetchall()
    cursor.close()
    connection.close()
    return render_template('company.html', data=data)

# search results page
@app.route('/search-results')
def find_jobs():
    sql_keywords = [
        "add",
        "alter",
        "delete",
        "all",
        "backup database",
        "between",
        "case",
        "like",
        "select",
        "company",
        "title",
        "link",
        "upper",
        "column",
        "create",
        "desc",
        "drop",
        "from",
        "set",
        "table",
        "jobs",
        "union",
        "values",
        "view",
        "where"
    ]
    string = request.args['find-feature']
    for word in string.split(" "):
        if word.lower() in sql_keywords:
            return "Sorry, you are not permitted to search for that job."
    connection = sql.connect('ml_jobs.db')
    cursor = connection.cursor()
    command = f"select company, title, link from jobs where upper(title) like upper(\"%{string}%\")"
    cursor.execute(command)
    data = cursor.fetchall()
    cursor.close()
    connection.close()
    return render_template('search-results.html', data=data, search=string)

# run in debug mode
if __name__ == '__main__':
    app.run(debug = True)
