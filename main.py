#!/usr/bin/python3.8
# Anmol Kapoor

print('Updating Database. May take upto 2 minutes depending on Internet connection')
# Part 1 #######################################################################
'''
Part 1 in this project focuses on initializing the database of city names in the
United States of America. This database is hosted on sqlite under a local server
and called "cities". Using sqlite3, I am able to secure a connection to my
previously-setup of my local sqlite3 database.

cities dbo:
|-- usa
     |--city, state

To access this database, run:
$ sqlite3 cities.db
'''
import sqlite3 as sql
import sys

def create_connection(db_file):
    conn = None

    try:
        conn = sql.connect(db_file)
        cur = conn.cursor()
    except sql.Error as e:
        print(e)

    return conn, cur

conn, cursor = create_connection('cities.db')

try:
    cursor.execute('create table usa ( city nvarchar(50), state nvarchar(50))')
    conn.commit()
except sql.Error:
    print("Cities database was already initialized. Moving on...")
    pass

# Part 2 #######################################################################
'''
Part 2 in this project focuses on populating the database of city names in the
United States of America. This database is hosted on sqlite under a local server
and called "cities". The source for the information provided to populate this
database was provided by the Center for Disease Control government organization
at https://www.cdc.gov/500cities/pdf/500-Cities-Listed-by-State.pdf. This PDF
was downloaded to the local folder as "500-Cities-Listed-by-State.pdf" and then
processed using the PyPDF2 module. Processing only occurs as long the cities
database is empty. Hence, this database will not be filled with copies.

cities dbo:
|-- usa
     |--city, state

To access this database in sqlcmd, run:
$ sqlite3 cities.db
'''
import PyPDF2 as pydf

cursor.execute("select count(*) from usa;")
items = cursor.fetchone()[0]

if items == 0:
    pdf = open('500-cities-listed-by-state.pdf', 'rb')
    reader = pydf.PdfFileReader(pdf)

    text = ''
    for page in reader.pages:
        if reader.getPageNumber(page) < 12:
            page_lines = page.extractText().split('\n')[4:-3]
        else:
            page_lines = page.extractText().split('\n')[4:-8]
        for line in page_lines:
            # looking at each line in the pdf
            # if first character is not a digit...
            if not line[0].isdigit():
                name = line
                # make sure that the names are properly formatted
                if any((character.isdigit() or character == "'") for character in name):
                    done = False
                    for charID in range(len(name)):
                        if done:
                            done = False
                            continue
                        if name[charID].isdigit():
                            name = name[:charID] + '\n'
                            break
                        if name[charID] == "'":
                            name = name[:charID] + "\'\'" + name[charID + 1:]
                            done = True
                # only case in which this happens
                elif "OregonPortland" == name:
                    text += "Oregon\n"
                    name = "Portland"
                text += name + '\n'
            else:
                text += '\n'

    # this part assumes you have [state, city, \n, city, \n, state, city, \n]
    lines = text.split('\n')
    cities = []
    states, state = [], None
    for index in range(len(lines)):
        if lines[index] != '' and lines[index + 1] != '':
            state = lines[index]
        elif lines[index] != '':
            cities.append(lines[index])
            states.append(state)

    command = "insert into usa(city, state) values "
    for i in range(len(cities)):
        command += "('" + cities[i] + "','" + states[i] + "')"
        if i != len(cities) - 1:
            command += ','
    cursor.execute(command)
    conn.commit()
else:
    print("Cities Database for the USA seems to have already been uploaded.")
    print("If you think something is wrong with the database, please do")
    print("run: 'drop table usa' in sqlcmd program, & run this program again.")
    print("\n\n")
    pass

cursor.close()
conn.close()
sys.exit()

## All below parts are unnecessary for the web version

# Part 3 #######################################################################
'''
Part 3 in this project accesses the spreadsheet at
docs.google.com/spreadsheets/d/1057brcM4eALpCzIQWLOM3C6mvXfoAGp8n8XnYJFzbTc/ to
populate the job_links list, which will hold the names of all companies in
consideration for a job and their job page. To ensure the safety and security of
this google spreadsheet, I have limited this spreadsheet to readonly for
outsiders.

job_links:
    (company's name, company's careers page)
'''
import pickle, os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

scope = ['https://www.googleapis.com/auth/spreadsheets.readonly']
spreadsheet_id = '1057brcM4eALpCzIQWLOM3C6mvXfoAGp8n8XnYJFzbTc'
range_name = 'Sheet1!A2:C'

creds = None

if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', scope)
        creds = flow.run_local_server(port = 0)
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)

service = build('sheets', 'v4', credentials=creds)

sheet = service.spreadsheets()
result = sheet.values().get(spreadsheetId = spreadsheet_id, range=range_name).execute()
values = result.get('values', [])

job_links = []

if not values:
    #print('No data found.')
    pass
else:
    print('Compiling and Updating the Job Repository Sources...\n\n\n')
    for row in values:
        job_links.append((row[0], row[2]))

# Part 4 #######################################################################
'''
Part 4 is the largest of all parts of this program. Using a combination of
Selenium and BeautifulSoup, I can scrape the url links as collected from part 3.
While requests was a popular module for use in web scraping, it is fundamentally
limited in processing the DOM code as created when incorporating the website's
JavaScript. Thus, Selenium with a headless Chrome is used to grab web elements
that require interaction to get a list of jobs, and a fast BeautifulSoup module
to search HTML pages, which works much faster than Selenium by itself. All is
composed into the job_listings table within the jobs sqlite database. To make
this process faster, I have utilized multiprocessing to run multiple processes
at once.

jobs
|--job_listings
    |--company name, job title, job application link
'''
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time, sys, unicodedata
#from multiprocessing import Pool
import logging, threading

job_conn, job_cursor = create_connection('jobs.db')

try:
    job_cursor.execute('create table job_listings (company nvarchar(20), title nvarchar(120), link nvarchar(175))')
    job_conn.commit()
except:
    pass

job_cursor.close()
job_conn.close()

job_conn, job_cursor = create_connection('jobs.db')

try:
    job_cursor.execute('create table to_rep (company nvarchar(20), title nvarchar(120), link nvarchar(175))')
    job_conn.commit()
except:
    pass

job_cursor.close()
job_conn.close()

def switch():
    job_conn, job_cursor = create_connection('jobs.db')

    job_cursor.execute('drop table job_listings')
    job_conn.commit()
    job_cursor.execute("alter table `to_rep` rename to `job_listings`")
    job_conn.commit()

print('I am starting to process the data from the job pages. Please wait a few')
print('moments for me to finish this task...\n\n\n')

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('log-level=3') # fatal errors only
options.add_argument('no-sandbox') # potentially solve tab crash problems
options.add_argument('disable-gpu') # potentially solve GpuChannelMsg_CreateCommandBuffer
options.add_argument('mute-audio') # to resolve issues with MediaEvents

job_opps = []

company_count = 0

def scrape(job_link):
    logging.info("Task Starting: %s", job_link[0])
    global job_opps, company_count
    retry_count = 2
    company, url = job_link
    company_count += 1
    browser = webdriver.Chrome(options=options)
    dbc = company.split(' ')[0]
    db = 'jobs/' + dbc + '.db'
    job_conn, job_cursor = create_connection(db)
    job_cursor.execute('create table co_jobs (company nvarchar(20), title nvarchar(120), link nvarchar(175))')
    job_conn.commit()

    browser.get(url)
    time.sleep(0.2)

    if 'hubspot' in url.lower():
        browser.find_element(By.CLASS_NAME, 'sc-kkGfuU').click()

    html = browser.page_source

    sp = BeautifulSoup(html, 'html.parser')

    #print(company + ': ', end = '')
    #print(company)
    if 'greenhouse' in url.lower():
        sections = sp.find_all('div', class_ = 'opening')
        for job in sections:
            title = unicodedata.normalize('NFKD', job.a.string).strip().replace("'", "")
            pre_link = job.a.get('href')
            if 'http' in pre_link:
                link = pre_link
            else:
                link = "/".join(url.split("/")[0:3]) + pre_link
            loc = unicodedata.normalize('NFKD', job.span.string).strip()
            job_opps.append((company, title, link))
            command = f"insert into co_jobs(company, title, link) values ('{company}', '{title}', '{link}')"
            job_cursor.execute(command)
            job_conn.commit()
        #print('\t\t\t', len(sections), 'job opportunities at {}!'.format(company))

    if 'ultipro' in url.lower():
        while True:
            load_more = browser.find_element(By.CSS_SELECTOR, "[data-bind='visible: skip() + pageSize < totalCount() && totalCount() > 0']")
            if load_more.get_attribute('style') == '':
                load_more.click()
                time.sleep(0.2)
            else:
                break
        sp = BeautifulSoup(browser.page_source, 'html.parser')
        sections = sp.find_all('div', class_='opportunity')
        for opp in sections:
            t_n_l = opp.find_all('a')[0]
            title = t_n_l.string
            loc = opp.find_all(attrs = {'data-bind' : 'text: Address().CityStatePostalCodeAndCountry()'})[0].string
            link = '/'.join(url.split('/')[0:3]) + t_n_l.get('href')
            job_opps.append((company, title, link))
            command = f"insert into co_jobs(company, title, link) values ('{company}', '{title}', '{link}')"
            job_cursor.execute(command)
            job_conn.commit()
            #more_locs = opp.find('small', attrs = {'data-automation' : 'job-location-more'})
            #if more_locs.string != "+0 more":
                #browser.get(link)
                #jl = BeautifulSoup(browser.page_source, 'html.parser')
                #jl_locs = list(dict.fromkeys(jl.find_all(attrs = {'data-bind' : 'text: Address().CityStatePostalCodeAndCountry()'})))
                #for new_place in jl_locs[1:]:
                    #additional_job_location = new_place.string
        #print(len(sections), 'job opportunities!', end = '')
        #print('\t\t\t', len(sections), 'job opportunities at {}!'.format(company))

    if 'hubspot' in url.lower():
        sections = sp.find_all('a', class_='sc-bdVaJa bKWNxX')
        for opp in sections:
            title = opp.find_all('p', class_ = 'sc-htpNat iUzPVU')[0].string
            loc = opp.find_all('p', class_ = 'sc-ifAKCX gHfmgn')[0].string
            link = '/'.join(url.split('/')[0:3]) + opp.get('href')
            job_opps.append((company, title, link))
            command = f"insert into co_jobs(company, title, link) values ('{company}', '{title}', '{link}')"
            job_cursor.execute(command)
            job_conn.commit()
        #print('\t\t\t', len(sections), 'job opportunities at {}!'.format(company))

    if 'powr' in url.lower():
        sections = sp.find_all('div', class_='row jobListing')
        #print(len(sections), 'job opportunities!', end = '')
        for opp in sections:
            if sections.index(opp) > 0:
                browser = webdriver.Chrome(options=options)
                browser.get(url)
            browser.execute_script("document.body.style.zoom='50%';")
            title = opp.find('div').find('div').find('h4').string.strip()
            loc = opp.find('div').find('div').find('p', class_ = 'details inline').string.split(' •')[0].strip()
            to_click = browser.find_elements(By.CLASS_NAME, 'more-info')[sections.index(opp)]
            browser.execute_script('arguments[0].click();', to_click)
            browser.implicitly_wait(7)
            time.sleep(0.1)
            application_btn = browser.find_elements(By.CLASS_NAME, 'btn')[-1]
            browser.execute_script('arguments[0].click();', application_btn)
            browser.implicitly_wait(7)
            time.sleep(0.1)
            link = browser.find_element(By.CLASS_NAME, 'apply.inline').get_attribute('href')
            job_opps.append((company, title, link))
            command = f"insert into co_jobs(company, title, link) values ('{company}', '{title}', '{link}')"
            job_cursor.execute(command)
            job_conn.commit()
            if opp != sections[-1]:
                browser.close()
                browser.quit()
        #print('\t\t\t', len(sections), 'job opportunities at {}!'.format(company))

    if 'lever' in url.lower():
        sections = sp.find_all('div', class_='posting')
        #print(len(sections), 'job opportunities!', end = '')
        for opp in sections:
            title = opp.find_all('h5')[0].string
            loc = opp.find_all('span', class_ = 'sort-by-location posting-category small-category-label')[0].string
            link = opp.find_all('a')[0].get('href')
            job_opps.append((company, title, link))
            command = f"insert into co_jobs(company, title, link) values ('{company}', '{title}', '{link}')"
            job_cursor.execute(command)
            job_conn.commit()
        #print('\t\t\t', len(sections), 'job opportunities at {}!'.format(company))

    if 'tripactions' in url.lower():
        sections = sp.find_all('li', class_='posting')
        #print(len(sections), 'job opportunities!', end = '')
        for opp in sections:
            title = opp.find_all('div', class_ = 'title')[0].string
            loc = opp.find_all('div', class_ = 'location')[0].string
            link = '/'.join(url.split('/')[0:3]) + opp.find_all('a')[0].string
            job_opps.append((company, title, link))
            command = f"insert into co_jobs(company, title, link) values ('{company}', '{title}', '{link}')"
            job_cursor.execute(command)
            job_conn.commit()
        #print('\t\t\t', len(sections), 'job opportunities at {}!'.format(company))

    if 'bird' in url.lower():
        sections = sp.find_all('div', class_='job-title')
        #print(len(sections), 'job opportunities!', end = '')
        for opp in sections:
            title = opp.find_all('span', class_ = 'job-meta strong')[0].string
            loc = opp.find_all('span', class_ = 'job-meta location')[0].string
            link = opp.find_all('a')[0].get('href')
            job_opps.append((company, title, link))
            command = f"insert into co_jobs(company, title, link) values ('{company}', '{title}', '{link}')"
            job_cursor.execute(command)
            job_conn.commit()
        #print('\t\t\t', len(sections), 'job opportunities at {}!'.format(company))

    if 'breezy' in url.lower():
        sections = sp.find_all('li', class_='position transition')
        #print(len(sections), 'job opportunities!', end = '')
        for opp in sections:
            title = opp.find_all('h2')[0].string
            loc = opp.find_all('li', class_ = 'location')[0].string
            link = "/".join(url.split("/")[0:3]) + opp.find_all('a')[0].get('href')
            job_opps.append((company, title, link))
            command = f"insert into co_jobs(company, title, link) values ('{company}', '{title}', '{link}')"
            job_cursor.execute(command)
            job_conn.commit()
        #print('\t\t\t', len(sections), 'job opportunities at {}!'.format(company))

    if 'scale' in url.lower():
        sections = sp.find_all('li', class_ = 'Jobs_itemWrapper__3u3uA bg-white py-2 px-4 rounded-1 shadow-md hover:shadow-xl transition-shadow duration-250 ease-out')
        #print(len(sections), 'job opportunities!', end = '')
        for opp in sections:
            title = opp.find_all('h3', class_ = 'font-normaexport default Jobs;l text-base text-black mb-2')[0].string
            loc = opp.find_all('div', class_ = 'font-normal text-sm text-gray-600 mb-2')[0].string
            link = '/'.join(url.split('/')[0:3]) + opp.find_all('a')[0].get('href')
            job_opps.append((company, title, link))
            command = f"insert into co_jobs(company, title, link) values ('{company}', '{title}', '{link}')"
            job_cursor.execute(command)
            job_conn.commit()
        #print('\t\t\t', len(sections), 'job opportunities at {}!'.format(company))

    if 'activision' in url.lower():
        job_count = 0
        while True:
            sp = BeautifulSoup(browser.page_source, 'html.parser')
            sections = sp.find_all('div', class_ = 'information')
            job_count += len(sections)
            for opp in sections:
                title = unicodedata.normalize('NFKD', "".join(str(opp.find_all('span')[0].a.div.span.string).split("'"))).strip().replace("'", "")
                loc = opp.find_all('p', class_ = 'job-info')[0].find_all('span')[1].string
                link = opp.find_all('a')[0].get('href').replace("'", "")
                job_opps.append((company, title, link))
                command = f"insert into co_jobs(company, title, link) values ('{company}', '{title}', '{link}')"
                job_cursor.execute(command)
                job_conn.commit()
            next_page = browser.find_elements(By.CSS_SELECTOR, "[aria-label='View next page']")[0]
            if next_page.get_attribute('href') != None:
                browser.execute_script('arguments[0].click();', next_page)
                time.sleep(0.2)
            else:
                break
        #print(job_count, 'job opportunities!', end = '')
        #print('\t\t\t', job_count, 'job opportunities at {}!'.format(company))

    if 'coursera' in url.lower():
        categories = browser.find_elements(By.CLASS_NAME, 'role')
        jobs = 0
        for cat_num in range(len(categories)):
            browser.execute_script('arguments[0].click();', categories[cat_num])
            browser.implicitly_wait(7)
            time.sleep(0.2)
            sp = BeautifulSoup(browser.find_element(By.CLASS_NAME, 'dept-roles-wrapper').get_attribute('outerHTML'), 'html.parser')
            sections = sp.find_all('a', class_='role-block')
            jobs += len(sections)
            for opp in sections:
                title = opp.find_all('h2')[0].string
                loc = opp.find_all('div')[0].string
                link = '/'.join(url.split('/')[0:3]) + opp.get('href')
                job_opps.append((company, title, link))
                command = f"insert into co_jobs(company, title, link) values ('{company}', '{title}', '{link}')"
                job_cursor.execute(command)
                job_conn.commit()
            browser.execute_script('arguments[0].click();', categories[cat_num])
            browser.implicitly_wait(7)
        #print(jobs, 'job opportunities!', end = '')
        #print('\t\t\t', jobs, 'job opportunities at {}!'.format(company))

    if 'coinbase' in url.lower():
        categories = browser.find_elements(By.CLASS_NAME, 'Department__Wrapper-sc-1n8uxi6-0.jItAmd')
        for category in categories:
            browser.execute_script('arguments[0].scrollIntoView(true);', category)
            browser.execute_script('window.scrollBy(0, -200);')
            browser.implicitly_wait(5)
            webdriver.ActionChains(browser).move_to_element(category).click(on_element = category).perform()
            browser.implicitly_wait(5)
            time.sleep(0.23)
        sp = BeautifulSoup(browser.find_element(By.CLASS_NAME, 'Positions__PositionsColumn-jve35q-7.jmaYDM').get_attribute('outerHTML'), 'html.parser')
        sections = sp.find_all('div', class_='Department__Job-sc-1n8uxi6-6 cgTJyi')
        for opp in sections:
            title = opp.find_all('a')[0].string
            loc = opp.find_all('div', class_ = 'Department__JobLocation-sc-1n8uxi6-8 iuVWuT')[0].string
            link = "/".join(url.split("/")[0:3]) + opp.find_all('a')[0].get('href')
            job_opps.append((company, title, link))
            command = f"insert into co_jobs(company, title, link) values ('{company}', '{title}', '{link}')"
            job_cursor.execute(command)
            job_conn.commit()
        #print(len(sections), 'job opportunities!', end = '')
        #print('\t\t\t', len(sections), 'job opportunities at {}!'.format(company))

    while True:
        try:
            if 'intuitive' in url.lower():
                to_click = browser.find_element(By.CSS_SELECTOR, '.mat-select-arrow.ng-tns-c64-33')
                browser.implicitly_wait(2.5)
                browser.execute_script('arguments[0].click();', to_click)
                browser.implicitly_wait(3)
                to_click2 = browser.find_elements(By.TAG_NAME, 'mat-option')[3]
                browser.execute_script('arguments[0].click();', to_click2)
                browser.implicitly_wait(10)
                jobs = 0
                while True:
                    section_tag = browser.find_element(By.TAG_NAME, 'mat-accordion')
                    sp = BeautifulSoup(section_tag.get_attribute('innerHTML'), 'html.parser')
                    sections = sp.find_all('mat-expansion-panel-header')
                    jobs += len(sections)
                    for section in sections:
                        title = section.contents[0].contents[0].contents[1].string
                        link = '/'.join(url.split('/')[0:3]) + section.contents[0].contents[0].contents[1].contents[0]['href']
                        loc = section.contents[0].contents[1].contents[0].contents[0].contents[0].contents[0].contents[1].string
                        job_opps.append((company, title, link))
                        command = f"insert into co_jobs(company, title, link) values ('{company}', '{title}', '{link}')"
                        job_cursor.execute(command)
                        job_conn.commit()
                    next_page = browser.find_element(By.CSS_SELECTOR, "[aria-label='Next Page of Job Search Results']")
                    if next_page.get_attribute('disabled') == None:
                        browser.execute_script('arguments[0].click();', next_page)
                        browser.implicitly_wait(15)
                        browser.refresh()
                        browser.implicitly_wait(4)
                    else:
                        break
                #print(jobs, 'job opportunities!', end = '')
                #print('\t\t\t', jobs, 'job opportunities at {}!'.format(company))
            break
        except KeyboardInterrupt:
            sys.exit()
        except:
            job_opps = [job for job in job_opps if job[0] != company]
            command = f"delete from job_listings where company = '{company}'"
            #print(command)
            job_cursor.execute(command)
            job_conn.commit()
            #print("command completed")
            if retry_count == 2:
                print(f"ERROR with '{company}'")
                print(sys.exc_info())
            retry_count -= 1
            if retry_count == 0:
                break
            #print('\n\nUmmm. This is embarrassing... Something went wrong, but I am')
            #print('going to try again right now. Please be patient, but feel')
            #print('free to restart the program as you wish.\n\n')

    if 'ttcportals' in url.lower():
        opportunities = 0
        page_count = len(sp.find_all('a', href = lambda href: href and '/jobs/search?page=' in href)) // 2
        for page in range(page_count):
            sp = BeautifulSoup(browser.page_source, 'html.parser')
            sections = [section for section in sp.find_all('div') if len(section.contents) > 0 and 'h3' in [sec.name for sec in section.contents]]
            for job in sections[1:]:
                title = job.find_all('h3')[0].find_all('a')[0].string
                link = job.find_all('h3')[0].find_all('a')[0]['href']
                loc = job.find_all('div')[0].find_all('div')[0].find_all('a')[0].string
                job_opps.append((company, title, link))
                command = f"insert into co_jobs(company, title, link) values ('{company}', '{title}', '{link}')"
                job_cursor.execute(command)
                job_conn.commit()
            opportunities += len(sections[1:])

            if page < page_count - 1:
                browser.find_element(By.CLASS_NAME, 'next_page').click()
                browser.implicitly_wait(10)

        job_opps = list(dict.fromkeys(job_opps))
        #print(opportunities, 'job opportunities!', end = '')
        #print('\t\t\t', opportunities, 'job opportunities at {}!'.format(company))

    while(True):
        try:
            if 'docusign' in url.lower():
                categs = browser.find_elements(By.CLASS_NAME, 'careers-fpp')
                times = len(categs)
                before = len(job_opps)
                for count in range(times):
                    category = browser.find_elements(By.CLASS_NAME, 'careers-fpp')[count]
                    dept_id = category.get_attribute('data-department-id')
                    to_click = browser.find_element(By.CSS_SELECTOR, "[data-department-id='" + dept_id + "']")
                    browser.implicitly_wait(2.5)
                    browser.execute_script('arguments[0].click();', to_click)
                    browser.implicitly_wait(10)
                    sp_ = BeautifulSoup(browser.page_source, 'html.parser')
                    table = sp_.find('table', class_ = 'job-table-' + dept_id)

                    for job in table.find_all('tr', class_ = 'tr-row'):
                        title = job.contents[0].a.string
                        link = job.contents[0].a['href']
                        loc = job.contents[1].string
                        job_opps.append((company, title, link))
                        command = f"insert into co_jobs(company, title, link) values ('{company}', '{title}', '{link}')"
                        job_cursor.execute(command)
                        job_conn.commit()

                    browser.refresh()
                    browser.implicitly_wait(1)
            else:
                break

            #print((len(job_opps) - before), 'job opportunities!', end = '')
            #print('\t\t\t', (len(job_opps) - before), 'job opportunities at {}!'.format(company))
            break
        except KeyboardInterrupt:
            sys.exit()
        except:
            job_opps = [job for job in job_opps if job[0] != company]
            command = f"delete from job_listings where company = '{company}'"
            #print(command)
            job_cursor.execute(command)
            job_conn.commit()
            #print("command completed")
            if retry_count == 2:
                print(f"ERROR with '{company}'")
                print(sys.exc_info())
            retry_count -= 1
            if retry_count == 0:
                break
            #print('\n\nUmmm. This is embarrassing... Something went wrong, but I am')
            #print('going to try again right now. Please be patient, but feel')
            #print('free to restart the program as you wish.\n\n')

    browser.close()
    browser.quit()

    job_cursor.close()
    job_conn.close()

    logging.info("Task Finishing: %s", company)

    #print()
#print('\n\n')

try:
    #start_time = time.time()
    form = "%(asctime)s: %(message)s"
    logging.basicConfig(format=form, level = logging.INFO, datefmt="%H:%M:%S")
    logging.info("Main : before creating thread")
    threads = [threading.Thread(target=scrape, args=(job_link,)) for job_link in job_links]
    logging.info("Main : before starting thread")
    for thr in threads:
        thr.start()
    for thr in threads:
        thr.join()
except:
    job_conn, job_cursor = create_connection('jobs.db')
    job_cursor.execute('drop table co_jobs')
    job_conn.commit()
    print("Error occurred. Please try again later.")

# job_conn, job_cursor = create_connection('jobs.db')

# switch()
#job_cursor.execute("select count(*) from job_listings")
#total_job_opportunities = job_cursor.fetchone()[0]
# command = "with cte as ( select company, title, link, ROW_NUMBER() over (partition by company, title, link order by company, title, link) row_num from job_listings ) delete from cte where row_num > 1"
# job_cursor.execute(command)
# job_conn.commit()
#job_cursor.execute("select count(*) from job_listings")
#final_job_opportunities = job_cursor.fetchone()[0]
#job_cursor.execute("select count(distinct company) from job_listings")
#total_company_count = job_cursor.fetchone()[0]

# job_cursor.close()
# job_conn.close()

#print("Time it took to load job opps from " + str(total_company_count) + " companies: {} seconds".format(time.time() - start_time))
#print("There are {} jobs available!".format(final_job_opportunities))
#print(f"There were {total_job_opportunities} job opportunities, but those {total_job_opportunities - final_job_opportunities} jobs seem to be duplicates.")


print('\n\nBased on the resources that I have on hand, I am ready to start')
print('displaying this information for you!')
