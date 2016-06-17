from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import timeit
import codecs

#browser = webdriver.Firefox()
browser = webdriver.PhantomJS()


queue_file = open("qeue.txt", "r")
while(True):
    myqueue_name = queue_file.readline()
    myqueue_url = queue_file.readline()
    myqueue_page_num = queue_file.readline()
    myqueue_name = myqueue_name.strip('\n')
    myqueue_url = myqueue_url.strip('\n')
    myqueue_page_num = myqueue_page_num.strip('\n')
    if(myqueue_name == ''):
        break

    file_name = myqueue_name
    myurl = myqueue_url
    total_page = int(myqueue_page_num)
    print(file_name)
    #################################################################################################################
    start = timeit.default_timer()
    browser.get(myurl)
    browser.refresh()
    corrent_page = 'page-1'
    #browser.implicitly_wait(30)
    page_number = 1
    mydata = []

    comment_file = open('collected_data/'+file_name+'.txt', 'wb')
    rate_file = open('collected_data/'+file_name+'_rates.txt', 'w')
    error_file = open("error_file.txt", "w")

    while(1):
        time.sleep(8)
        print("page number is: ", page_number, " of ", total_page)
        html_source = browser.page_source
        soup = BeautifulSoup(html_source, 'html.parser')
        comments = soup.findAll('div',{'class':'comment-text'})
        user_ratings = soup.findAll('div',{'class':'user-rating'})

        mytry = 0
        while(len(comments) == 0 or len(user_ratings) == 0):
            mytry += 1
            if(mytry == 10):
                error_file.write(file_name)
                error_file.write('\n')
                print('error with this phone')
                break
            browser.get(myurl)
            browser.refresh()
            time.sleep(2)
            html_source = browser.page_source
            soup = BeautifulSoup(html_source, 'html.parser')
            comments = soup.findAll('div', {'class': 'comment-text'})
            user_ratings = soup.findAll('div', {'class': 'user-rating'})
            print("mytry is: ", mytry)

        for i, c in enumerate(comments):
            user_rate = user_ratings[i]
            user_rate = user_rate.contents[1]

            scores = []
            k = 1
            while(k <= 11):
                the_rate = user_rate.contents[k]
                k += 2
                the_rate = the_rate.contents[3]

                counter = 0
                j = 1
                while(j <= 9):
                    tmp = the_rate.contents[j]
                    j += 2
                    tmp = str(tmp)
                    if("done" in tmp):
                        counter += 1
                scores.append(counter)

            s = str(c.contents[1])
            s = BeautifulSoup(s, 'lxml').get_text()
            s = s.replace('\n', '. ').replace('\r', '. ')
            s = s.replace('  ',' ')
            s = s.strip()
            #print(s)
            comment_file.write(s.encode("UTF-8"))
            comment_file.write('\n'.encode("UTF-8"))
            score_string = '['
            for the_count, entery in enumerate(scores):
                score_string = score_string + str(entery)
                if((the_count +1) < len(scores)):
                    score_string = score_string + ', '
                else:
                    score_string = score_string + ']'
            rate_file.write(score_string)
            rate_file.write('\n')
            mydata.append(s)
        print("page" + str(page_number) + " done")
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

    comment_file.close()
    rate_file.close()
    stop = timeit.default_timer()
    spent_time = int(stop - start)
    sec = int(spent_time % 60)
    spent_time = spent_time / 60
    minute = int(spent_time % 60)
    spent_time = spent_time / 60
    hours = int(spent_time)
    print("h: " + str(hours) + "  minutes: " + str(minute) + "  secunds: " + str(sec))

browser.quit()
queue_file.close()
error_file.close()
wait = input("PRESS ENTER TO CONTINUE.")
print("Finish")