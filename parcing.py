import requests
from bs4 import BeautifulSoup
import csv
import os
import time

URL = 'https://auto.ria.com/uk/newauto/marka-jeep/'
HEADERS = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36',
    'accept': '*/*'
}
HOST = 'https://auto.ria.com'
FILE = 'cars.csv'


def get_html(url, params=None):
    r = requests.get(url, headers=HEADERS, params=params)
    return r

def get_pages_count(html):
    soup = BeautifulSoup(html, 'html.parser')
    pagination = soup.find_all('span', class_='mhide')
    if pagination:
        return int(pagination[-1].get_text())
    else:
        return 1


def get_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    items = soup.find_all('a', class_='proposition_link')

    cars = []
    for item in items:
        # из-за того что иногда цена в гривне не указана, сделаем проверку её наличия
        uah_price = item.find('span', class_='size16')
        if uah_price:
            uah_price = uah_price.get_text()
        else:
            uah_price = 'Цену уточняйте'
        cars.append({
            'title': item.find('div', class_='proposition_title').get_text(strip=True),
            'link': HOST + item.get('href'),
            'usd_price': item.find('span', class_='green').get_text(),
            'uah_price': uah_price,
            'city': item.find('span', class_='item region').get_text(strip=True)
        })

    return cars


def save_file(items, path):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Марка', 'Ссылка', 'Цена в $', 'Цена в UAH', 'Город'])
        for item in items:
            writer.writerow([item['title'], item['link'], item['usd_price'], item['uah_price'], item['city']])


def parse():
    URL = input('Введите URL: ')
    URL = URL.strip()
    html = get_html(URL)
    if html.status_code == 200:
        cars = []
        pages_count = get_pages_count(html.text)
        for page in range(1, pages_count+1):
            print('Парсинг страницы {} из {}...'.format(page, pages_count))
            html = get_html(URL, params={'page': page})
            cars.extend(get_content(html.text))
            time.sleep(2)
        save_file(cars, FILE)
        print('Получено {} автомобилей'.format(len(cars)))
        os.startfile(FILE)
    else:
        print('Error')


parse()
