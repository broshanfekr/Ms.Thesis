__author__ = 'BeRo'
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import timeit
import codecs


#browser = webdriver.Firefox()
browser = webdriver.PhantomJS()

myurl = ''
browser.get(myurl)
#browser.refresh()
html_source = browser.page_source
soup = BeautifulSoup(html_source, 'html.parser')
phones = soup.findAll('div',{'class':'pimg'})

myQlist = open('qeue.txt', 'w')
for l in phones:
    #l = phones[2]
    x = str(l).split('"')
    #print(x[3])
    next_url = "http://www.digikala.com" + x[3] + "#!/displaycomment-0/page-1/sort-date/tab-comments/"
    
    browser.get(next_url)
    browser.refresh()
    time.sleep(5)
    html_source = browser.page_source
    soup = BeautifulSoup(html_source, 'html.parser')
    phone_name = soup.findAll('h1',{'itemprop':'name'})
    phone_name = phone_name[0].contents[0]
    phone_name = phone_name.strip()
    phone_name = phone_name.replace('/', ' ')
    phone_name = phone_name.replace("\\", ' ')
    page_count = soup.findAll('div',{'class':'dk-pagination-container light-theme simple-pagination'})
    if(len(page_count) == 0):
        page_count = 1
    else:
        page_count = page_count[0].contents[0]
        page_count = page_count.contents[-2]
        page_count = page_count.contents[0]
        page_count = page_count.contents[0]

    print(phone_name)
    print(next_url)
    print(page_count)
    myQlist.write(phone_name)
    myQlist.write('\n')
    myQlist.write(next_url)
    myQlist.write('\n')
    myQlist.write(str(page_count))
    myQlist.write('\n')


myQlist.close()
wait = input("PRESS ENTER TO CONTINUE.")