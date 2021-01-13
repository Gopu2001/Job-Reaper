# Anmol Kapoor

# Step 1: Enter everything into the database
import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import sqlite3 as sql
import sys, os, time
import pandas as pd

# create a list of stopwords as found on https://www.ranks.nl/stopwords
stopwords = ['a', 'an', 'the', 'at', 'about', 'around', 'as', 'below', 'by',
             'up', 'above', 'for', 'in', 'into', 'like', 'near', 'of', 'off',
             'on', 'onto', 'outside', 'over', 'past', 'per', 'round', 'through',
             'to', 'toward', 'towards', 'under', 'underneath', 'until', 'via',
             'versus', 'vs', 'with']

# source for variety of jobs
sp = BeautifulSoup(requests.get("https://www.joblist.com/b/all-jobs").text, 'html.parser')

# get all elements with job titles
departments = sp.find_all("ul", class_="css-3dvgnl")[2:-1]

titles = []
for dept in departments:
    for a_tag in dept.find_all("a"):
        titles.append(a_tag.text)

# define a method for creating a connection to giant.db
# I rarely get an error message, so I haven't really refined the except block
def get_cnxn(src):
    conn, curs = None, None
    try:
        conn = sql.connect(src)
        curs = conn.cursor()
    except sql.Error as e:
        print(e)
        if os.path.exists(src):
            curs.close()
            conn.close()
        sys.exit(1)
    return conn, curs

## empty the database if it already exists to repopulate it
if os.path.exists("giant.db"):
    os.remove("giant.db")
# create the connection and the table in the db
conn, curs = get_cnxn("giant.db")
curs.execute('create table giant_jobs (title nvarchar(120), yes_no bit)')
conn.commit()

# insert all data from joblist.com
insertion_cmd = "insert into giant_jobs(title, yes_no) values "
last = False
for title in titles:
    titler = title.lower()
    rl_title = ""
    for word in (''.join([i for i in titler if i.isalpha() or i == ' '])).split(" "):
        if word not in stopwords:
            rl_title += word + " "
    rl_title = rl_title.strip()
    if title == titles[-1]:
        last = True
    insertion_cmd += f"('{rl_title}', '{1}')"
    if not last:
        insertion_cmd += ", "
curs.execute(insertion_cmd)
conn.commit()

# insert all data from output_non_jobs.txt
# this data is from hubspot's home page
# TODO: Get this to get all the text from the HubSpot page dynamically
with open("output_non_jobs.txt", mode="r", encoding="utf-8") as file:
    content = file.readlines()

insertion_cmd = "insert into giant_jobs(title, yes_no) values "
last = False
for line in content:
    liner = line.lower()
    rl_title = ""
    for word in (''.join([i for i in liner.replace("'", "") if i.isalpha() or i == ' '])).split(" "):
        if word not in stopwords:
            rl_title += word + " "
    rl_title = rl_title.strip()
    if line == content[-1]:
        last = True
    insertion_cmd += f"('{rl_title}', '{0}')"
    if not last:
        insertion_cmd += ", "
curs.execute(insertion_cmd)
conn.commit()

# Here we will add locations as false data, because apparently, we are getting that
cit_cn, cit_cu = get_cnxn("../cities.db")
cit_cu.execute("select distinct city from usa")
cities = [city[0] for city in cit_cu.fetchall()]
cit_cu.execute("select distinct state from usa")
states = [state[0] for state in cit_cu.fetchall()]
cities.append("Remote") # because of COVID situation

# An updated cities database was made available through SimpleMaps
# Please head to https://simplemaps.com/data/world-cities and download the free
# version as a CSV for the updated file.
# NOTE: maintain the filename as worldcities.csv
world_cities_db = pd.read_csv("worldcities.csv")
world_cities = list(set([city for city in world_cities_db["city"].tolist() if city not in cities and city not in states]))
world_states = list(set([state for state in world_cities_db["admin_name"].tolist() if state not in world_cities and state not in cities and state not in states]))
world_countries = list(set([country for country in world_cities_db["country"].tolist()]))

cities += world_cities
states += world_states
countries = world_countries

## handle the cities first
insertion_cmd = "insert into giant_jobs(title, yes_no) values "
last = False
for city in cities:
    cityr = city.lower()
    rl_title = ""
    for word in (''.join([i for i in cityr.replace("'", "") if i.isalpha() or i == ' '])).split(" "):
        if word not in stopwords:
            rl_title += word + " "
    rl_title = rl_title.strip()
    if city == cities[-1]:
        last = True
    insertion_cmd += f"('{rl_title}', '{0}')"
    if not last:
        insertion_cmd += ", "
curs.execute(insertion_cmd)
conn.commit()

## handle the states now
insertion_cmd = "insert into giant_jobs(title, yes_no) values "
last = False
for state in states:
    if type(state) != str: # sometimes there is the occasional nan bc of worldcities.csv
        continue
    stater = state.lower()
    rl_title = ""
    for word in (''.join([i for i in stater.replace("'", "") if i.isalpha() or i == ' '])).split(" "):
        if word not in stopwords:
            rl_title += word + " "
    rl_title = rl_title.strip()
    if state == states[-1]:
        last = True
    insertion_cmd += f"('{rl_title}', '{0}')"
    if not last:
        insertion_cmd += ", "
curs.execute(insertion_cmd)
conn.commit()

## handle the countries now
insertion_cmd = "insert into giant_jobs(title, yes_no) values "
last = False
for country in countries:
    countryr = country.lower()
    rl_title = ""
    for word in (''.join([i for i in countryr.replace("'", "") if i.isalpha() or i == ' '])).split(" "):
        if word not in stopwords:
            rl_title += word + " "
    rl_title = rl_title.strip()
    if country == countries[-1]:
        last = True
    insertion_cmd += f"('{rl_title}', '{0}')"
    if not last:
        insertion_cmd += ", "
curs.execute(insertion_cmd)
conn.commit()

cit_cu.close()
cit_cn.close()

# Create a list of common languages
common_languages = [
    "English",
    "Mandarin",
    "Chinese",
    "Spanish",
    "Hindi",
    "Arabic",
    "Portuguese",
    "Bengali",
    "Russian",
    "Japanese",
    "Punjabi",
    "Javanese",
    "German",
    "Korean",
    "French",
    "Telugu",
    "Marathi",
    "Turkish",
    "Tamil",
    "Vietnamese",
    "Urdu",
    "Indonesian"
]

## handle the languages now
insertion_cmd = "insert into giant_jobs(title, yes_no) values "
last = False
for lang in common_languages:
    langr = lang.lower()
    rl_title = ""
    for word in (''.join([i for i in langr.replace("'", "") if i.isalpha() or i == ' '])).split(" "):
        if word not in stopwords:
            rl_title += word + " "
    rl_title = rl_title.strip()
    if lang == common_languages[-1]:
        last = True
    insertion_cmd += f"('{rl_title}', '{0}')"
    if not last:
        insertion_cmd += ", "
curs.execute(insertion_cmd)
conn.commit()

# Here we will add job titles as true data, because apparently, we are getting too many positives
job_cn, job_cu = get_cnxn("../jobs.db")
job_cu.execute("select distinct title from jobs")
jobs = [job[0] for job in job_cu.fetchall()]

## handle the jobs first
insertion_cmd = "insert into giant_jobs(title, yes_no) values "
last = False
for job in jobs:
    jobr = job.lower()
    rl_title = ""
    for word in (''.join([i for i in jobr.replace("'", "") if i.isalpha() or i == ' '])).split(" "):
        if word not in stopwords:
            rl_title += word + " "
    rl_title = rl_title.strip()
    if job == jobs[-1]:
        last = True
    insertion_cmd += f"('{rl_title}', '{1}')"
    if not last:
        insertion_cmd += ", "
curs.execute(insertion_cmd)
conn.commit()

job_cu.close()
job_cn.close()

# now add some phrases that are common on job sites, but are not job titles
general_terms = [
    "Department",
    "All Departments",
    "Office",
    "All Offices",
    "Finance",
    "Current Job Openings",
    "Global Support",
    "Human Resources",
    "IT",
    "Marketing",
    "Product",
    "Product Designer",
    "Professional Services",
    "Research and Development",
    "Engineering",
    "Applications",
    "Cloud Engineering",
    "Data Infrastructure & Security",
    "Data Platforms",
    "Quality & Release",
    "Security",
    "Runtime",
    "SQL",
    "Sales",
    "Alliances",
    "Corporate Sales",
    "Customer & Product Strategy",
    "Sales Engineering",
    "Sales Operations",
    "Workplace",
    "Agriculture, Food, & Natural Resources",
	"Architecture & Construction",
	"Arts, Audio/Video Technology, and Communications",
	"Business, Management, & Administration",
	"Education & Training",
	"Government & Public Administration",
	"Health Science",
	"Hospitality & Tourism",
	"Information Technology",
	"Law, Public Safety, Corrections, & Security",
	"Manufacturing",
	"Marketing, Sales, & Service",
	"Science, Technology, Engineering, & Mathematics",
    "Technology",
	"Transportation, Distribution, & Logistics",
    "Entry-level",
    "Entry-Level",
    "Entry Level",
    "Entry level",
    "Associate",
    "Mid-Senior level",
    "Mid-Senior-level",
    "Mid-Senior Level",
    "Mid-Senior-Level",
    "Director",
    "Executive",
    "Senior"
]

# Here we will add related to job titles as false data, because apparently, we are getting too many positives
insertion_cmd = "insert into giant_jobs(title, yes_no) values "
last = False
for phrase in general_terms:
    phraser = phrase.lower()
    rl_title = ""
    for word in (''.join([i for i in phraser.replace("'", "") if i.isalpha() or i == ' '])).split(" "):
        if word not in stopwords:
            rl_title += word + " "
    rl_title = rl_title.strip()
    if phrase == general_terms[-1]:
        last = True
    insertion_cmd += f"('{rl_title}', '{0}')"
    if not last:
        insertion_cmd += ", "
curs.execute(insertion_cmd)
conn.commit()

def load_from_web(website):
    # Now we will get all the text from [Website]'s homepage
    soup = BeautifulSoup(requests.get(website, allow_redirects=False).text, 'html.parser')
    text = [string.strip() for string in soup.get_text().split("\n") if string.strip() != "" and "<" not in string and any(char.isalpha() for char in string)]

    # Here we will add all the text from [Website]'s homepage as false data
    insertion_cmd = "insert into giant_jobs(title, yes_no) values "
    last = False
    for phrase in text:
        phraser = phrase.lower()
        rl_title = ""
        for word in (''.join([i for i in phraser.replace("'", "") if i.isalpha() or i == ' '])).split(" "):
            if word not in stopwords:
                rl_title += word + " "
        rl_title = rl_title.strip()
        if rl_title == "":
            continue
        if phrase == text[-1]:
            last = True
        insertion_cmd += f"('{rl_title}', '{0}')"
        # if not last:
        insertion_cmd += ", "
    insertion_cmd = insertion_cmd[0:-2]
    try:
        curs.execute(insertion_cmd)
        conn.commit()
    except:
        print(insertion_cmd)
        sys.exit(1)

websites = ["https://www.activision.com/",
            "https://www.valent.com/",
            "https://www.gensler.com/",
            "https://www.prada.com/us/en.html",
            "https://www.bain.com/",
            "https://teachable.com/",
            "https://www.jpmorganchase.com/",
            "https://www.whitehouse.gov/",
            "https://www.pfizer.com/",
            "https://www.modernatx.com/",
            "https://www.expedia.com/",
            "https://www.collabera.com/",
            "http://www.peoplebusiness.org/",
            "https://www.fbi.gov/",
            "https://www.sbsheriff.org/command-and-divisions/custody-operations/jail-operations-division/jail-facilities/",
            "https://www.thorindustries.com/",
            "https://www.vectormarketing.com/",
            "https://www.llnl.gov/",
            "https://www.ups.com/us/en/Home.page",
            "https://www.docusign.com/"]#,
            # "https://www.usps.com/"] # This doesn't work because it errors out with max retries or gives ''

for website in websites:
    print(website)
    load_from_web(website)
    time.sleep(1)

# finally close the connection to giant.db
curs.close()
conn.close()
