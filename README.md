# Jobbing-GUI Extension

Jobbing-GUI Extension, also known as Jobbing-GUI Version E, is a Python application meant to keep a track on jobs from a specific set of companies. The list of companies that were used for this project can be found on this [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1057brcM4eALpCzIQWLOM3C6mvXfoAGp8n8XnYJFzbTc/). If you are unable to click on the link, you can copy and paste the following into a new tab: https://docs.google.com/spreadsheets/d/1057brcM4eALpCzIQWLOM3C6mvXfoAGp8n8XnYJFzbTc/.

The inspiration for this project came from my search for jobs. There already exist tools like LinkedIn and Indeed that display a lot of jobs, however, these tools notify you of job openings from thousands of companies, many of which you may choose not to work with.

<!-- ## Installation

A public version of this project has not yet been released, but if you wish, you may clone this repository, and start keeping track of jobs by running main.py like so:

```bash
python3.8 main.py
```

This will take approximately 1 minute to gather all of the data.

The web and desktop interfaces will be available soon. -->

## Installation

Please pip install the following packages. Note that the version of Python used is 3.8.

```bash
pip install pysqlite3 PyPDF2 pydf selenium bs4 flask flask_restful flask_sqlalchemy pandas sklearn
```
For installing PyTorch, you will need to go to [the PyTorch Website](https://pytorch.org/get-started/locally/) to figure out the command.
For instance, if installing using pip on cpu, run:
```bash
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```
You will need to install the chromedriver for your specific Chrome version. Please go to: https://chromedriver.chromium.org/. If you install the latest chromedriver, please be sure to make sure that your Chrome is updated to the most recent version.

You may need to install sqlite3 if you have not already.

## Running the code

In navigating this project, keep in mind the following:
* run ```python3 main.py``` to populate the cities.db
* run ```python3 web.py``` to run the flask server
* run ```python3 scrape_struct.py``` to populate ml_jobs.db (which is used by web.py)
* you will require a credentials.json file to run scrape_struct.py. This is to connect with the Google Sheets API and will be provided upon request.
* All html files are templates which are used with web.py
* run ```python3 NLP_Practice\job_classifier.py``` to train the BERT model
* run ```python3 'NLP_Practice\naiveBayes jobs.py'``` to train the Naive Bayes model
* run ```python3 GiantDB\giant.py``` to populate giant.db for use with training the models listed above
* In the case you get the error where "saved_weights.pt" file cannot be found, you will need to run ```cd NLP_Practice && python3 job_classifier.py && cd ..``` to train the Bert model. Due to the model weights taking just under 0.5 GB of storage space, this file cannot be uploaded.
* You may notice some errors with can't find file or related errors as some organization took place after so as to declutter the main folder. Contact me if you have any questions with finding any file.
=======
# Jobbing-GUI Extension

Jobbing-GUI Extension, also known as Jobbing-GUI Version E, is a Python application meant to keep a track on jobs from a specific set of companies. The list of companies that were used for this project can be found on this [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1057brcM4eALpCzIQWLOM3C6mvXfoAGp8n8XnYJFzbTc/). If you are unable to click on the link, you can copy and paste the following into a new tab: https://docs.google.com/spreadsheets/d/1057brcM4eALpCzIQWLOM3C6mvXfoAGp8n8XnYJFzbTc/.

The inspiration for this project came from my search for jobs. There already exist tools like LinkedIn and Indeed that display a lot of jobs, however, these tools notify you of job openings from thousands of companies, many of which you may choose not to work with.

<!-- ## Installation

A public version of this project has not yet been released, but if you wish, you may clone this repository, and start keeping track of jobs by running main.py like so:

```bash
python3.8 main.py
```

This will take approximately 1 minute to gather all of the data.

The web and desktop interfaces will be available soon. -->

## Installation

Please pip install the following packages. Note that the version of Python used is 3.8.

```bash
pip install pysqlite3 PyPDF2 pydf google-api-python-client google-auth-httplib2 google-auth-oauthlib selenium bs4 flask flask_restful flask_sqlalchemy pandas sklearn
```
For installing PyTorch, you will need to go to [the PyTorch Website](https://pytorch.org/get-started/locally/) to figure out the command.
For instance, if installing using pip on cpu, run:
```bash
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```
You will need to install the chromedriver for your specific Chrome version. Please go to: https://chromedriver.chromium.org/. If you install the latest chromedriver, please be sure to make sure that your Chrome is updated to the most recent version.

You may need to install sqlite3 if you have not already. To install sqlite3, please go to https://www.sqlite.org/download.html and scroll down to "Precompiled Binaries for [YOUR OS]" and click to install the zipped file. IMPORTANT: FOR WINDOWS USERS: Under "Precompiled Binaries for Windows", please click "sqlite-tools-win32-x86-[LATEST VERSION].zip" to install the command-line shell program. After downloading these files, you may place the "sqlite3.exe" file in the project folder, or to install the file universally, you may store this file into a system directory and ensure that that directory is added to the System PATH.

## Running the code

In navigating this project, keep in mind the following:
* run ```python3 main.py``` to populate the cities.db
* run ```python3 web.py``` to run the flask server
* run ```python3 scrape_struct.py``` to populate ml_jobs.db (which is used by web.py)
* you will require a credentials.json file to run scrape_struct.py. This is to connect with the Google Sheets API and will be provided upon request.
* All html files are templates which are used with web.py
* run ```python3 NLP_Practice\job_classifier.py``` to train the BERT model
* run ```python3 'NLP_Practice\naiveBayes jobs.py'``` to train the Naive Bayes model
* run ```python3 GiantDB\giant.py``` to populate giant.db for use with training the models listed above
* You may notice some errors with can't find file or related errors as some organization took place after so as to declutter the main folder. Contact me if you have any questions with finding any file.

## Other Files
This section will be updated in the Wiki soon.
=======
# Jobbing-GUI Extension

Jobbing-GUI Extension, also known as Jobbing-GUI Version E, is a Python application meant to keep a track on jobs from a specific set of companies. The list of companies that were used for this project can be found on this [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1057brcM4eALpCzIQWLOM3C6mvXfoAGp8n8XnYJFzbTc/). If you are unable to click on the link, you can copy and paste the following into a new tab: https://docs.google.com/spreadsheets/d/1057brcM4eALpCzIQWLOM3C6mvXfoAGp8n8XnYJFzbTc/.

The inspiration for this project came from my search for jobs. There already exist tools like LinkedIn and Indeed that display a lot of jobs, however, these tools notify you of job openings from thousands of companies, many of which you may choose not to work with.

<!-- ## Installation

A public version of this project has not yet been released, but if you wish, you may clone this repository, and start keeping track of jobs by running main.py like so:

```bash
python3.8 main.py
```

This will take approximately 1 minute to gather all of the data.

The web and desktop interfaces will be available soon. -->

## Installation

Please pip install the following packages. Note that the version of Python used is 3.8.

```bash
pip install pysqlite3 PyPDF2 pydf google-api-python-client google-auth-httplib2 google-auth-oauthlib selenium bs4 flask flask_restful flask_sqlalchemy pandas sklearn
```
For installing PyTorch, you will need to go to [the PyTorch Website](https://pytorch.org/get-started/locally/) to figure out the command.
For instance, if installing using pip on cpu, run:
```bash
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```
You will need to install the chromedriver for your specific Chrome version. Please go to: https://chromedriver.chromium.org/. If you install the latest chromedriver, please be sure to make sure that your Chrome is updated to the most recent version.

You may need to install sqlite3 if you have not already. To install sqlite3, please go to https://www.sqlite.org/download.html and scroll down to "Precompiled Binaries for [YOUR OS]" and click to install the zipped file. IMPORTANT: FOR WINDOWS USERS: Under "Precompiled Binaries for Windows", please click "sqlite-tools-win32-x86-[LATEST VERSION].zip" to install the command-line shell program. After downloading these files, you may place the "sqlite3.exe" file in the project folder, or to install the file universally, you may store this file into a system directory and ensure that that directory is added to the System PATH.

## Running the code

In navigating this project, keep in mind the following:
* run ```python3 main.py``` to populate the cities.db
* run ```python3 web.py``` to run the flask server
* run ```python3 scrape_struct.py``` to populate ml_jobs.db (which is used by web.py)
* you will require a credentials.json file to run scrape_struct.py. This is to connect with the Google Sheets API and will be provided upon request.
* All html files are templates which are used with web.py
* run ```python3 NLP_Practice\job_classifier.py``` to train the BERT model
* run ```python3 'NLP_Practice\naiveBayes jobs.py'``` to train the Naive Bayes model
* run ```python3 GiantDB\giant.py``` to populate giant.db for use with training the models listed above
* You may notice some errors with can't find file or related errors as some organization took place after so as to declutter the main folder. Contact me if you have any questions with finding any file.

## Other Files
This section will be updated in the Wiki soon.
