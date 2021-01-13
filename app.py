# Part 5 #######################################################################
'''
Part 5 is the final part of the project. It will display all of the results into
a tkinter window. after clicking a company name from the main page, you may
click a job title that you are interested in and a web browser will
automatically open with the application link for that application.
'''
# IMPORTS
import tkinter as tk
import sys, pyodbc, webbrowser

# INIT TOP WINDOW
top = tk.Tk(className = ' Job Post Listings ')
w, h = 500, 525
top.geometry(str(w) + 'x' + str(h))
top_status = 0
top.resizable(False, False)

# INIT QUERY WINDOW
query_win = tk.Tk(className = ' Submit a Job Query ')
query_win.withdraw()
qw, qh = 300, 50
query_win.geometry(str(qw + 16) + 'x' + str(qh))
query_win.resizable(False, False)
query_string = None
q_string = "%%"

# INIT GLOBAL VARIABLES
under_frame = tk.Frame(top, bd = -2, bg = "white")
under_frame.pack(fill = tk.BOTH, expand = True)
under_frame.columnconfigure(0, minsize = w // 2)
under_frame.columnconfigure(1, minsize = w // 2)
mouse = ('man', 'pencil')
font = ('Courier', 11)
comp_page = 0
LIMIT = 20
companies = []
jobs = []

# SWITCH TO MAIN SCREEN, KILL PROGRAM
# PROGRAM AUTOMATICALLY ENDS WHEN TOP+QUERY WINDOWS ARE DESTROYED
def end_state(event):
    if top_status == 0:
        top.destroy()
        sys.exit()
    elif top_status == 1:
        start()

# DESTROY QUERY WINDOW
def close_query(event):
    query_win.destroy()

# OPEN THE JOB APPLICATION LINK IN DEFAULT BROWSER
def open_job_link(link):
    webbrowser.open(link)

# CLEAR ALL WIDGETS FOR NO TRANFERENCE
def clear_mem():
    global jobs, companies
    for job in jobs:
        job.grid_forget()
        job.destroy()
    jobs = []
    for company in companies:
        company.grid_forget()
        company.destroy()
    companies = []

# SETUP QUERY WINDOW WITH SUBMIT BTN
def open_query():
    global query_string
    query_win.deiconify()
    for widget in get_children(query_win):
        widget.grid_forget()
    tk.Label(query_win, text = "Job Title: ", anchor='w').grid(column = 0, row = 0)
    query_string = tk.Entry(query_win, width = 20)
    query_string.grid(column = 1, row = 0)
    query_string.bind('<Return>', lambda event : run_query())
    submit_query = tk.Button(query_win, text = "Submit", command = run_query)
    submit_query.grid(column = 2, row = 0)

# RUN A QUERY FROM SUBMIT QUERY BUTTON AND RELOAD TOP WINDOW
def run_query():
    global query_string, q_string
    q_string = "%" + query_string.get() + "%"
    start()

# DISPLAY ALL JOBS FOR SPECIFIC COMPANY WITH SCROLLING WINDOW
def company_jobs(company_name):
    global top_status, jobs
    top_status = 1
    clear_mem()
    minimum = False

    job_conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};Server=localhost;Database=jobs;UID=SA;PWD=Apno0227')
    job_cursor = job_conn.cursor()
    command = f"select title, link from job_listings where company = '{company_name}' and title like '{q_string}' order by title"
    job_cursor.execute(command)
    job_opps = job_cursor.fetchall()
    job_cursor.close()
    job_conn.close()
    
    for job in job_opps:
        job_button = tk.Button(under_frame, text = job[0], anchor = 'w', font = font, height = 2, width = 53, command = lambda link = job[1]: open_job_link(link))
        jobs.append(job_button)

    add_jobs(page_num = 0)

    for widget in get_children(top):
        widget['cursor'] = mouse[top_status]

def add_jobs(page_num = 0):
    for i in range(10):
        under_frame.rowconfigure(i, minsize = 50)
    under_frame.rowconfigure(i+1, minsize = 20)
    for btn in get_children(under_frame):
        btn.grid_forget()
    last = 10*(page_num+1)
    if len(jobs) < last:
        last = len(jobs)

    if page_num == 0:
        command = None
    else:
        command = lambda num = page_num-1: add_jobs(page_num = num)
    left = tk.Button(under_frame, text = '<', font = font, width = 25, command = command)

    print()
    if last == len(jobs):
        command = None
    else:
        command = lambda num = page_num+1: add_jobs(page_num = num)
    right = tk.Button(under_frame, text = '>', font = font, width = 25, command = command)
    left.grid(row = 10, column = 0, columnspan = 1, sticky = tk.W)
    right.grid(row = 10, column = 1, columnspan = 1, sticky = tk.W)
    
    row = 0
    for job_button in jobs[10*page_num : last]:
        job_button.grid(row = row, column = 0, columnspan = 2, sticky = tk.W)
        row += 1

# DISPLAY 'X' BY 2 COMPANY BUTTON LIST WITH NUMBER OF JOBS
# BUTTON COMMAND TO DISPLAY JOBS FOR THAT COMPANY
def start():
    global top_status, comp_page, LIMIT, companies
    top_status = 0
    clear_mem()

    col_index = 0
    row = 0
    company_add_count = 0

    job_conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};Server=localhost;Database=jobs;UID=SA;PWD=Apno0227')
    job_cursor = job_conn.cursor()
    job_cursor.execute('select distinct company from job_listings order by company')
    company_names_list = job_cursor.fetchall()
    
    for company_name in company_names_list:
        company_add_count += 1
        command = f"select count(title) from job_listings where company = '{company_name[0]}' and title like '{q_string}'"
        job_cursor.execute(command)
        numJobs = job_cursor.fetchone()[0]
        if numJobs == 0:
            continue
        company_button = tk.Button(under_frame, text = company_name[0] + '\n' + str(numJobs) + ' Jobs Available', font = font, width = 25, command = lambda s = company_name[0]: company_jobs(s))
        companies.append(company_button)

    add_companies(page_num = comp_page)

    for widget in get_children(top):
        widget['cursor'] = mouse[top_status]

def add_companies(page_num = 0):
    global comp_page
    comp_page = page_num
    for i in range(10):
        under_frame.rowconfigure(i, minsize = 50)
    under_frame.rowconfigure(i+1, minsize = 20)
    for btn in get_children(under_frame):
        btn.grid_forget()
    last = 20*(page_num+1)
    if len(companies) < last:
        last = len(companies)

    if page_num == 0:
        command = None
    else:
        command = lambda num = page_num-1: add_companies(page_num = num)
    left = tk.Button(under_frame, text = '<', font = font, width = 25, command = command)

    if last == len(companies):
        command = None
    else:
        command = lambda num = page_num+1: add_companies(page_num = num)
    right = tk.Button(under_frame, text = '>', font = font, width = 25, command = command)
    left.grid(row = 10, column = 0, sticky = tk.W)
    right.grid(row = 10, column = 1, sticky = tk.W)
    
    row = 0
    col = 0
    for company_button in companies[20*page_num : last]:
        company_button.grid(row = row, column = col, sticky = tk.W)
        if col == 1:
            row += 1
        col = (col + 1) % 2

# GET ALL CHILDREN IN A WIDGET TO CHANGE MOUSE LOOK FASTER
def get_children(widget):
    widget_list = widget.winfo_children()
    for wid in widget_list:
        if wid.winfo_children():
            widget_list.extend(get_children(wid))
    return widget_list

# START APP WINDOWS
if __name__ == '__main__':
    top.bind('<Escape>', end_state)
    top.bind_all('<Control-Key-f>', lambda event : open_query())
    query_win.bind('<Escape>', close_query)
    start()
    #button_canvas.bind_all('<Button-4>', lambda event: button_canvas.yview_scroll(-1, "units"))
    #button_canvas.bind_all('<Button-5>', lambda event: button_canvas.yview_scroll( 1, 'units'))
    tk.mainloop()
