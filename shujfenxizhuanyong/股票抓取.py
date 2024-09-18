import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

headers = {'user-agent':'Mozilla/5.0 \
            (Windows NT 10.0; Win64; x64) \
            AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/84.0.4147.105 Safari/537.36'}

urls = [
    'https://groww.in/us-stocks/nke',
    'https://groww.in/us-stocks/ko', 
    'https://groww.in/us-stocks/msft', 
    'https://groww.in/stocks/m-india-ltd', 
    'https://groww.in/us-stocks/axp', 
    'https://groww.in/us-stocks/amgn', 
    'https://groww.in/us-stocks/aapl', 
    'https://groww.in/us-stocks/ba', 
    'https://groww.in/us-stocks/csco', 
    'https://groww.in/us-stocks/gs', 
    'https://groww.in/us-stocks/ibm', 
    'https://groww.in/us-stocks/intc', 
    'https://groww.in/us-stocks/jpm', 
    'https://groww.in/us-stocks/mcd',
    'https://groww.in/us-stocks/crm', 
    'https://groww.in/us-stocks/vz', 
    'https://groww.in/us-stocks/v', 
    'https://groww.in/us-stocks/wmt',  
    'https://groww.in/us-stocks/dis'
    ]

all=[]
for url in urls:
    page = requests.get(url,headers=headers)
    try:
        soup = BeautifulSoup(page.text, 'html.parser')
        company = soup.find('h1', {'class': 'usph14Head displaySmall'}).text
        price = soup.find('span', {'class': 'uht141Pri contentPrimary displayBase'}).text
        change = soup.find('div', {'class': 'uht141Day bodyBaseHeavy contentNegative'}).text
        volume=soup.find('table', {'class': 'tb10Table col l5'}).find_all('td')[1].text
        x=[company,price,change,volume]
        all.append(x)
        
    except AttributeError:
      print("Change the Element id")
    # Wait for a short time to avoid rate limiting
    time.sleep(10)

column_names = ["Company", "Price", "Change","Volume"]
df = pd.DataFrame(columns = column_names)
for i in all:
  index=0
  df.loc[index] = i
  df.index = df.index + 1
df=df.reset_index(drop=True)
df.to_excel('stocks.xlsx')
