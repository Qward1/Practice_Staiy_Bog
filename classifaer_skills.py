import re
from itertools import chain
import sqlite3
from nltk import ngrams
from pymorphy3 import MorphAnalyzer

morph = MorphAnalyzer()


def normalize_text(text):
    # приводим к именительному падежу и нормализуем
    def normalize_word(word):
        parsed = morph.parse(word)[0]
        return parsed.normalized.word.lower()

    # удалеем знаки и разбивка на слова
    words = re.findall(r'\b[а-яa-z]+\b', text.lower())
    # Нормализуем каждое слово
    return [normalize_word(word) for word in words]


def prepare_skills(skills_dict):
    # словарь нормализованных навыков и их синонимов
    normalized_skills = {}
    for main_skill, synonyms in skills_dict.items():
        # нормализуем основной навык
        main_words = normalize_text(main_skill)
        main_key = ' '.join(main_words)

        if not isinstance(synonyms, list):
            synonyms = [synonyms]

        # нормализуем все синонимы
        # пока на обдумывание , как лучше реализовать
        all_variants = [main_skill] + synonyms

        for variant in all_variants:
            variant_words = normalize_text(variant)
            ngram_variants = []

            # создаем n граммы для варианта
            for n in range(1, 4):
                ngram_variants += [' '.join(gram) for gram in ngrams(variant_words, n)]

            # добавляем в словарь
            for ngram in ngram_variants:
                if ngram not in normalized_skills:
                    normalized_skills[ngram] = main_key
    return normalized_skills


def extract_skills(text, skills_dict):
    # нормализуем текст
    normalized_words = normalize_text(text)

    # создаем n граммы
    text_ngrams = []
    for n in range(1, 4):
        text_ngrams += [' '.join(gram) for gram in ngrams(normalized_words, n)]


    skills_mapping = prepare_skills(skills_dict)


    # ищем совпадения
    found_skills = set()
    for ngram in text_ngrams:
        if ngram in skills_mapping:
            found_skills.add(skills_mapping[ngram])


    return list(found_skills)


def process_vacancies(db_path, skills_dict):
    # конектимся к базе данных
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # подготавливаем словарь навыков one раз
    skills_mapping = prepare_skills(skills_dict)

    # вытаскиваем все вакансии
    cursor.execute('SELECT link, description FROM vacancies')
    vacancies = cursor.fetchall()

    # обрабатываем вакансии
    for link, description in vacancies:
        if not description:
            skills_str = ''
        else:
            # извлекаем навыки
            skills_list = extract_skills(description, skills_mapping)
            skills_str = ', '.join(skills_list)

        # сохраняем в бдшку
        cursor.execute('''
            INSERT OR REPLACE INTO vacancy_skills (link, skills)
            VALUES (?, ?)
        ''', (link, skills_str))

    # комитим и закрываем
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Словарь навыков и синонимов (ключ - основной навык, значение - список синонимов) сделан нейронкой))
    # но в свои словари в ручную делать будем скорее всего
    skills = {
        "Python": [
            # Основные названия и синонимы
            "Питон", "Пайтон", "Python3", "Python 3", "Python Developer",
            "Python Engineer", "Python программист", "Разработчик Python",
            "Специалист Python", "Пайтек", "Пайтонист", "Python-разработка",

            # Фреймворки и библиотеки
            "Django", "Flask", "FastAPI", "Pyramid", "Bottle", "Falcon",
            "Tornado", "aiohttp", "Sanic", "CherryPy", "Dash", "Streamlit",

            # ORM и базы данных
            "SQLAlchemy", "Django ORM", "Peewee", "Pony ORM", "Psycopg2",
            "SQLObject", "MongoEngine", "Redis-py",

            # Асинхронное программирование
            "Asyncio", "AsyncIO", "Асинхронное программирование",
            "Асинхронный Python", "aiohttp", "aiofiles",

            # Тестирование
            "Pytest", "Unittest", "Doctest", "Nose", "Tox", "Hypothesis",
            "Selenium", "Locust", "Allure",

            # Data Science и ML
            "Pandas", "NumPy", "SciPy", "Scikit-learn", "TensorFlow", "PyTorch",
            "Keras", "OpenCV", "NLTK", "Spacy", "Matplotlib", "Seaborn", "Plotly",
            "Dask", "PySpark", "XGBoost", "LightGBM", "CatBoost",

            # Веб и API
            "REST API", "GraphQL", "WebSocket", "JSON-RPC", "SOAP",
            "Django REST Framework", "DRF", "Flask-RESTful", "FastAPI",
            "Swagger", "OpenAPI",

            # Инструменты и утилиты
            "Jupyter", "IPython", "PyCharm", "VS Code", "Virtualenv", "Pipenv",
            "Poetry", "Conda", "Docker", "Celery", "RabbitMQ", "Kafka",
            "Elasticsearch", "Sentry", "Prometheus",

            # Системы контроля версий и CI/CD
            "Git", "GitHub", "GitLab", "Bitbucket", "Jenkins", "GitHub Actions",
            "GitLab CI", "TeamCity", "Travis CI",

            # Шаблоны и методологии
            "ООП", "SOLID", "Design Patterns", "Микросервисы", "MVC",
            "Функциональное программирование", "TDD", "BDD", "DDD",

            # Операционные системы
            "Linux", "Unix", "Bash", "Shell scripting",

            # Облачные платформы
            "AWS", "Lambda", "EC2", "S3", "Google Cloud", "Azure",
            "Heroku", "DigitalOcean", "Kubernetes", "Docker Swarm",

            # Смежные технологии
            "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "ClickHouse",
            "HTML", "CSS", "JavaScript", "TypeScript", "React", "Vue.js",

            # Задачи и процессы
            "Парсинг данных", "Веб-скрейпинг", "Автоматизация", "Скриптинг",
            "Оптимизация производительности", "Рефакторинг", "Code Review",
            "Профилирование", "Дебаггинг", "Юнит-тесты", "Интеграционные тесты",
            "Деплоймент", "DevOps", "Agile", "Scrum", "Kanban",

            # Форматы данных
            "JSON", "XML", "YAML", "CSV", "Protobuf", "Avro",

            # Безопасность
            "JWT", "OAuth", "SSL/TLS", "Хэширование", "Шифрование",
            "SQL инъекции", "XSS защита",

            # Уровни разработки
            "Backend разработка", "Full-stack разработка", "API разработка",
            "Микросервисная архитектура", "Высоконагруженные системы",
            "Распределенные системы"
        ],

        # Другие навыки (примеры)
        "JavaScript": ["JS", "ECMAScript", "ES6+", "Node.js", "TypeScript"],
        "Машинное обучение": ["ML", "ИИ", "Искусственный интеллект", "Нейронные сети"],
        "Базы данных": ["SQL", "NoSQL", "Реляционные БД", "Нереляционные БД"],
        "DevOps": ["CI/CD", "Infrastructure as Code", "Terraform", "Ansible"],
        "Техническое лидерство": ["Team Lead", "Архитектор", "Менторство"]
    }

    process_vacancies('db_for_vacancies.sqlite3', skills)

    conn = sqlite3.connect('db_for_vacancies.sqlite3')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM vacancy_skills LIMIT 5')
    print('\nПервые пять штук: ')
    for row in cursor.fetchall():
        print(f"Ссылка: {row[0]}\nНавыки: {row[1]}")
    conn.close()
