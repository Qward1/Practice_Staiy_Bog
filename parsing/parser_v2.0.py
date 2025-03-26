import requests
import sqlite3
import random
import time
import logging


hh_api_token = 'APPLO2LFALFT1KA1RTCR8LANBL6GQ15C6SU6V92NF5H3M9590IJ24DM15QVR8RFJ'


DB_FILE = '/db_for_vacancies.sqlite3'


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_connection():
    """Создание подключения к базе данных SQLite"""
    return sqlite3.connect(DB_FILE)

# создание таблицы vacancies
def create_table(conn):
    cursor = conn.cursor()

    create_table_query = """
        CREATE TABLE IF NOT EXISTS vacancies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            company TEXT,
            industry TEXT,
            title TEXT,
            keywords TEXT,
            skills TEXT,
            experience TEXT,
            salary TEXT,
            url TEXT UNIQUE
        )
    """
    cursor.execute(create_table_query)
    conn.commit()
    cursor.close()
    logging.info("Таблица 'vacancies' успешно создана.")

# удаление таблицы vacancies
def drop_table(conn):
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS vacancies_mobile_dev")
    conn.commit()
    cursor.close()
    logging.info("Таблица 'vacancies' успешно удалена.")

# получение вакансий
def get_vacancies(vacancy, page):
    url = 'https://api.hh.ru/vacancies'
    params = {
        'text': f"{vacancy}",
        'specialization': 1,
        'per_page': 100,
        'page': page
    }
    headers = {
        'Authorization': f'Bearer {hh_api_token}'
    }

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()

# получение скилов
def get_vacancy_skills(vacancy_id):
    url = f'https://api.hh.ru/vacancies/{vacancy_id}'
    headers = {
        'Authorization': f'Bearer {hh_api_token}'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    skills = [skill['name'] for skill in data.get('key_skills', [])]
    return ', '.join(skills)

# парсинг вакансий
def parse_vacancies():
    cities = {
        'Москва': 1,
        'Санк_Петербург': 2,

    }

    vacancies = [

    # Нужные нам ключевые слова


    ]

    conn = create_connection()
    drop_table(conn)
    create_table(conn)

    try:
        for city, city_id in cities.items():
            for vacancy in vacancies:
                page = 0
                while True:
                    try:
                        data = get_vacancies(vacancy, page)

                        if not data.get('items'):
                            break

                        cursor = conn.cursor()
                        for item in data['items']:
                            if vacancy.lower() not in item['name'].lower():

                                continue

                            title = f"{item['name']} ({city})"
                            keywords = item['snippet'].get('requirement', '')
                            skills = get_vacancy_skills(item['id'])
                            company = item['employer']['name']
                            industry = item['employer'].get('industry', {}).get('name', '')
                            experience = item['experience'].get('name', '')
                            salary = item.get('salary', {})
                            salary_str = "з/п не указана"
                            if salary:
                                salary_from = salary.get('from', '')
                                salary_to = salary.get('to', '')
                                salary_str = f"{salary_from}-{salary_to}" if salary_from or salary_to else "з/п не указана"
                            url = item['alternate_url']

                            insert_query = """
                                INSERT OR IGNORE INTO vacancies
                                (city, company, industry, title, keywords, skills, experience, salary, url) 
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """
                            cursor.execute(insert_query, (
                                city, company, industry, title, keywords,
                                skills, experience, salary_str, url
                            ))

                        conn.commit()
                        cursor.close()

                        if page >= data['pages'] - 1:
                            break

                        page += 1
                        time.sleep(random.uniform(3, 6))

                    except requests.HTTPError as e:
                        logging.error(f"Ошибка при обработке города {city}: {e}")
                        continue

    finally:
        conn.close()

    logging.info("Парсинг завершен. Данные сохранены в базе данных SQLite.")

# для удаления дубликатов
def remove_duplicates():
    conn = create_connection()
    try:
        cursor = conn.cursor()
        conn.commit()
    finally:
        conn.close()
    logging.info('Дубликаты в таблице "vacancies" успешно обработаны.')

def run_parsing_job():
    logging.info("Запуск парсинга...")
    try:
        parse_vacancies()
        remove_duplicates()
    except Exception as e:
        logging.error(f"Ошибка при выполнении задачи парсинга: {e}")


if __name__ == "__main__":
    run_parsing_job()
