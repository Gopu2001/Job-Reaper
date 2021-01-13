import re, time, sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from bs4 import element

'''
This class is meant to be used with Jobbing GUI.
This can scrape a well-structured website by
looking for known equally structured data as
compared to a given set of data.
'''
class Website:
    '''
    Initialize global and local variables

    url:   url of the website to scrape
    '''
    def __init__(self, url=None, company=None):
        self.__url = url # The URL for the website to scrape.
        # only do the time-lengthy stuff when needed
        if self.__url:
            self.set_url(url)
        # local variable necessary for the get_by_path method
        self.link = None
        self.past_links = []
        self.company = company

    '''
    Getters and Setters for the url (string)
    '''
    def get_url(self):
        return self.__url
    def set_url(self, url):
        # setting up the webdriver
        self.CrOptions = webdriver.ChromeOptions()
        self.CrOptions.add_argument('headless')
        self.CrOptions.add_argument('log-level=3') # fatal errors only
        self.CrOptions.add_experimental_option('excludeSwitches', ['enable-logging']) #in attempt to get rid of the "devtools" line
        #self.CrOptions.add_argument('no-sandbox') # potentially solve tab crash problems
        #self.CrOptions.add_argument('disable-gpu') # potentially solve GpuChannelMsg_CreateCommandBuffer
        self.CrOptions.add_argument('mute-audio') # to resolve issues with MediaEvents
        # setting up the browser and getting the url post-JS loading
        self.browser = webdriver.Chrome(options=self.CrOptions, service_log_path="NUL")
        self.browser.get(url)
        time.sleep(1)
        page = 1
        self.html = self.browser.page_source
        # creating a soup with the html as the ingredient
        self.soup = BeautifulSoup(self.html, "html.parser")
        # inserting body tag's innerHTML from all the rest of the pages (2 to n)
        while(str(page+1) in self.soup.body.stripped_strings):
            # click on the element with just page#
            to_click = self.browser.find_elements(By.XPATH, f"//*[text()='{page+1}']")[0]
            self.browser.execute_script('arguments[0].click();', to_click)
            time.sleep(0.5)
            # updating the html and soup variables
            self.html = self.html.split("</body>")[0] + self.browser.find_element(By.TAG_NAME, "body").get_attribute('innerHTML') + "</body>" + self.html.split("</body>")[1]
            self.soup = BeautifulSoup(self.html, "html.parser")
            page += 1
            sys.stdout.write("\r")
            sys.stdout.write(f"Page: {page}")
            sys.stdout.flush()
        sys.stdout.write("\r")
        # Closing and quiting the browser's processes
        self.browser.close()
        self.browser.quit()

    '''
    Getters and Setters for the HTML (string)
    '''
    def get_html(self):
        return self.html
    def set_html(self, html):
        self.html = html

    '''
    Getter for the text in an HTML
    '''
    def get_text(self):
        return list(set([string for string in self.soup.body.stripped_strings]))
        # return [string.strip() for string in self.soup.body.get_text().split("\n") if string.strip() != "" and not "<" in string]

    '''
    Getter for the BeautifulSoup object pertaining to the data in this website
    '''
    def get_soup(self):
        return self.soup

    '''
    Return the count of tags by tag name.

    # Not necessary and may be deleted in the future
    '''
    def count_tags(self, tag, start=0, end=-1):
        return self.html[start:end].count("<"+tag)

    '''
    Given a string that is located in the web page, return the tag name path taken
    to get to that string.

    Sample return:
        body.div.div.section.a
    matcher: the string for which to get a navigable path
    '''
    def get_path(self, matcher):
        # for some reason, sometimes I get a None error
        if matcher not in self.html:
            return None
        # also return none if the number of words exceeds 5
        # so as to remove some garbage
        if len(matcher.split(" ")) > 5:
            return None
        # replace all characters that would not compile with regex so that re
        # will take every character as literals
        replaceable_chars = "\\ [ ] ( ) { } * + ? | ^ $ .".split(" ")
        # print(matcher, end=" -> ")
        for char in replaceable_chars:
            matcher = matcher.replace(char, "\\" + char)
        # print(matcher)
        elems = self.soup.find_all(text=re.compile(matcher))
        if len(elems) == 0:
            return None
        path = ""
        # print(matcher, self.soup.find_all(text=re.compile(matcher)))
        for elem in elems:
            for parent in elem.parents:
                path = parent.name + "." + path
                if parent.name == "body":
                    break
            if 'body' != path[0:4]:
                path = ""
            else:
                break
        return path[0:-1]

    '''
    Given a path, find all strings with a tag name path the same as the path

    path: path returned by get_path method
    current: default None (changes to body tag), the element from which to look at.
    '''
    def get_by_path(self, path, current=None):
        if path == None:
            return None
        path = path.split(".")
        matches = []
        if len(path) == 0:
            matches.append("Completed Recursion")
            return matches
        # find current
        if not current:
            current = self.soup.find("body")
        if len(current.contents) > 0:
            for child in current.contents:
                if child.name == "a":
                    try:
                        self.link = child["href"]
                    except KeyError:
                        self.link = None
                        # if the a tag doesn't have a href attribute, then assume no link
                if len(path) == 1 and type(child) == element.NavigableString:
                    if self.link != None and self.link != "" and self.link not in self.past_links and self.link != "javascript:void(0);" and self.link != "javascript:void(0)" and self.link[0] != "#" and str(child).strip() != "": ## assumption: there are href links
                        matches.append((child, self.link))
                        self.past_links.append(self.link)
                    ## print error message
                    # if self.link == None:
                    #     print("Error: Unable to get the application link for", str(child).strip() + ".")
                    #     print("Error: Unable to add", str(child).strip() + ".")
                        self.link = None
                elif len(path) == 1 and type(child) != element.NavigableString:
                    continue
                elif child.name == path[1]:
                    for string in self.get_by_path(".".join(path[1:]), child):
                        matches.append(string)
        return matches

if __name__ == "__main__":
    url = "https://www.hubspot.com/careers/jobs?page=1"
    company = "HubSpot"
    # sample = "Executive Assistant"
    # setup_start = time.time()
    webpage = Website(url, company)
    # print(webpage.get_text())
    # start = time.time()
    # path = webpage.get_path(sample)
    # all_commons = []
    # if path:
    #     print("I found the sample job! Looking for more jobs!")
    #     all_commons = webpage.get_by_path(path)
    #     end = time.time()
    #     # Print at most the first 10 results of the page (less if there are less # jobs found)
    #     length = len(all_commons)
    #     if length > 10:
    #         length = 10
    #     print("I have listed below the first", length, "course(s).\n")
    #     i = 1
    #     for job in all_commons[:length]:
    #         print(i, job[0])
    #         i += 1
    # else:
    #     print("No record of a", sample, "job in this job board.")
    #     end = time.time()
    # print("\n\nIt took", (end-start), "second(s) to search for the jobs!")
    # print("I found", len(all_commons), "jobs from the company:", webpage.company + ".")
    # print((start-setup_start), "seconds to connect to the webpage on Chrome.")
