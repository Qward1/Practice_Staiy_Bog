import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

CLIENT_ID = "OLSCPBIR884U8BK0QBI2QL5BUG32STAREP6SRK59H9VH78LDBU92I31PB8NNQ287"
CLIENT_SECRET = "KQ2GQ8IH9MN18NHP3DP2TAG0HG2UG46THOQ4DIQ0I6JB7G7D65QRJ2DSU76D80KA"
REDIRECT_URI = "https://example.com/page"
USER_AGENT = "ForPractica (45bgkxzwt2@privaterelay.appleid.com)"


def get_access_token():
    auth_url = (
        f"https://hh.ru/oauth/authorize?"
        f"response_type=code&"
        f"client_id={CLIENT_ID}&"
        f"redirect_uri={REDIRECT_URI}"
    )
    print(f"Перейдите по ссылке и авторизуйтесь:\n{auth_url}")
    code = input("Введите code из URL после авторизации: ")

    token_url = "https://hh.ru/oauth/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "redirect_uri": REDIRECT_URI
    }

    response = requests.post(token_url, data=data)
    if response.status_code != 200:
        raise Exception(f"Ошибка авторизации: {response.text}")

    return response.json()["access_token"]


def remove_html_tags(html_text):
    if not html_text:
        return ""

    soup = BeautifulSoup(html_text, 'html.parser')
    return soup.get_text(separator=' ', strip=True)


def fetch_vacancies(access_token, pages=5):
    base_url = "https://api.hh.ru/vacancies"
    headers = {"Authorization": f"Bearer {access_token}"}
    all_vacancies = []
    total_found = 0
    max_api_limit = 2000

    initial_response = requests.get(base_url, headers=headers, params={
        "text": "Python",
        "per_page": 1,
        "page": 0
    })

    if initial_response.status_code != 200:
        raise Exception(f"Ошибка начального запроса: {initial_response.text}")

    initial_data = initial_response.json()
    total_found = initial_data.get("found", 0)

    pages_needed = 99
    if total_found > max_api_limit:
        print(f"\n⚠️ Внимание: API HH.ru показывает только первые {max_api_limit} вакансий")
        total_found = max_api_limit

    with tqdm(total=pages_needed, desc="🔄 Загрузка страниц", unit="page") as pbar:
        for page in range(pages_needed):
            time.sleep(1)

            response = requests.get(base_url, headers=headers, params={
                "text": "Python",
                "per_page": 100,
                "page": page
            })

            if response.status_code != 200:
                print(f"\n❌ Ошибка на странице {page}: {response.text}")
                break

            data = response.json()
            all_vacancies.extend(data.get("items", []))
            pbar.update(1)

            if len(all_vacancies) >= total_found:
                break

    return {"items": all_vacancies[:total_found], "found": total_found}


if __name__ == "__main__":
    try:
        access_token = get_access_token()
        print("\n🔑 Успешная авторизация!")

        base_url = "https://api.hh.ru/vacancies"
        headers = {"Authorization": f"Bearer {access_token}"}

        vacancies_data = fetch_vacancies(access_token)

        print(f"\n📊 Всего найдено вакансий: {vacancies_data['found']}")
        print(f"💾 Загружено вакансий: {len(vacancies_data['items'])}")

        with tqdm(vacancies_data["items"], desc="📥 Обработка вакансий", unit="vac") as pbar:
            for idx, vacancy in enumerate(pbar):
                if idx > 0 and idx % 10 == 0:
                    time.sleep(2)
                print("\n" + "=" * 50)
                print(f"Должность: {vacancy['name']}")
                print(f"Компания: {vacancy['employer']['name']}")

                vacancy_id = vacancy["id"]
                full_vacancy = requests.get(f"{base_url}/{vacancy_id}", headers=headers).json()

                full_description = full_vacancy.get("description", "")
                clean_text = remove_html_tags(full_description)
                print(f"Полное описание вакансии {vacancy_id}:\n{clean_text}\n")

    except Exception as e:
        print(f"\n❌ Критическая ошибка: {str(e)}")