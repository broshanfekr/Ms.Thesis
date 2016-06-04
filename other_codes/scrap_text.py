from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time

total_page = 52
myurl = ''
#browser = webdriver.Firefox()
browser = webdriver.PhantomJS()
browser.get(myurl)
corrent_page = 'page-1'
#browser.implicitly_wait(30)

page_number = 1
mydata = []

myfile = open("collected_data.txt", 'w')
while(1):
    time.sleep(3)
    print("page number is: ", page_number)
    html_source = browser.page_source
    soup = BeautifulSoup(html_source, 'html.parser')
    comments = soup.findAll('div',{'class':'comment-text'})
    mytry = 0
    while(len(comments) == 0):
        mytry += 1
        if(mytry == 10):
            break
        browser.get(myurl)
        browser.refresh()
        time.sleep(2)
        html_source = browser.page_source
        soup = BeautifulSoup(html_source, 'html.parser')
        comments = soup.findAll('div', {'class': 'comment-text'})
    for c in comments:
        s = str(c.contents[1])
        s = BeautifulSoup(s, 'lxml').get_text()
        s = s.replace('\n', '. ').replace('\r', '. ')
        s = s.replace('  ',' ')
        print(s)
        myfile.write(s)
        myfile.write('\n')
        mydata.append(s)

    #browser.quit()
    #browser.close()
    if(page_number == total_page):
        break
    else:
        page_number += 1
        new_page = 'page-' + str(page_number)
        myurl = myurl.replace(corrent_page, new_page)
        corrent_page = new_page
        print(myurl)
        browser.get(myurl)
        browser.refresh()
        #browser = webdriver.Firefox()
        #browser = webdriver.PhantomJS()


browser.quit()
