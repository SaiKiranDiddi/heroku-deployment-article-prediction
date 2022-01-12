#from selenium import webdriver
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import date, timedelta
import multiprocessing

def scrap_article(URL):
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all("div", {"class": "article"})
    try:
        title = results[0].find("h1", {"class": "title"}).text
        content = results[0].find("div", {"id": re.compile(r"^content\-body.*")})
        content = content.find_all("p")
        content_text = ""
        for line in content:
            content_text += line.text
        return (title, content_text)
    except:
        print(URL)
        return (None, None)
    


def scrap_with_date(date):
    URL = "https://www.thehindu.com/archive/web/"+date+"/"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all("section", {"id": re.compile(r"^section.*")})
    # categories = []
    category_links = []
    for result in results:
        category = result.find("a", {"class": "section-list-heading"}).text.strip()
        links_section = result.find_all("li")
        rows = []
        for i in links_section[1:]:
            url_link = i.find("a")['href']
            title, content = scrap_article(url_link)
            rows += [[date,category,url_link,title,content]]
        category_links += rows
    return category_links

def func(month):
    start_date = date(2020, month, 1)
    end_date = date(2020, month, 2)
    data_list = []
    for i in range((end_date - start_date).days):
        curr_date = (start_date+i*timedelta(days=1)).strftime("%Y/%m/%d")
        print(curr_date)
        data_list += scrap_with_date(curr_date)
    data = pd.DataFrame(data_list, columns=['date', 'category', 'url', 'title', 'content'])
    data.to_csv('data_2021_'+str(month)+'_full.csv')
    print("Done for month "+str(month))


if __name__=="__main__":
    start_date = date(2013, 12, 1)
    end_date = date(2013, 12, 30)
    delta = timedelta(days=1)
    data_list = []
    for i in range((end_date - start_date).days):
        curr_date = (start_date+i*timedelta(days=1)).strftime("%Y/%m/%d")
        print(curr_date)
        data_list += scrap_with_date(curr_date)
    data = pd.DataFrame(data_list, columns=['date', 'category', 'url', 'title', 'content'])
    data.to_csv('data_2013_12_full.csv')
    # pool_obj = multiprocessing.Pool()
    # pool_obj.map(func,range(1,12))
        
        
        
