import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def load_and_preprocess_data(market_skills_path, university_skills_path):
    """
    Загружаем и предобрабатываем данные о навыках с рынка труда и университетов.

    Параметры:
        market_skills_path: путь к файлу с навыками рынка труда
        university_skills_path: путь к файлу с навыками университета

    Результат выполнения:
        market_skills_df: DataFrame с навыками рынка труда и их весами
        university_skills: список навыков университета
    """
    # загружвкм csv с рынка вакансий
    market_skills = pd.read_csv(market_skills_path, header=None)[0].tolist()

    # подсчет каждого навыка(коэффицент важности) , возможно будем менять
    skill_counter = Counter(market_skills)


    market_skills_df = pd.DataFrame({
        'skill': list(skill_counter.keys()),
        'weight': list(skill_counter.values())
    })

    # scaler
    max_weight = market_skills_df['weight'].max()
    market_skills_df['normalized_weight'] = market_skills_df['weight'] / max_weight

    # навыки с универов
    university_skills = pd.read_csv(university_skills_path, header=None)[0].tolist()

    return market_skills_df, university_skills


def analyze_skill_gap(market_skills_df, university_skills):
    """
    Анализ разрыва между навыками рынка труда и универов.

    Параметры:
        market_skills_df: DataFrame с навыками рынка труда и их весами
        university_skills: список навыков университета

    Результат выполнения:
        coverage: процент покрытия навыков рынка труда университетом
        missing_skills_df: DataFrame с навыками, отсутствующими в университетской программе
        overlap_skills_df: DataFrame с навыками, присутствующими и на рынке, и в университете
    """

    university_skills_set = set(university_skills)

    # определяем какие навыки рынка труда присутствуют/отсутствуют в университете
    market_skills_df['in_university'] = market_skills_df['skill'].apply(
        lambda x: x in university_skills_set
    )

    # навыки которые отсутствуют в университете
    missing_skills_df = market_skills_df[~market_skills_df['in_university']].sort_values(
        by='normalized_weight', ascending=False
    ).reset_index(drop=True)

    # навыки которые присутствуют и на рынке и в университете
    overlap_skills_df = market_skills_df[market_skills_df['in_university']].sort_values(
        by='normalized_weight', ascending=False
    ).reset_index(drop=True)

    # взвешенный процент покрытия навыков
    total_weight = market_skills_df['normalized_weight'].sum()
    covered_weight = overlap_skills_df['normalized_weight'].sum()

    coverage = (covered_weight / total_weight) * 100 if total_weight > 0 else 0

    return coverage, missing_skills_df, overlap_skills_df


def calculate_similarity(market_skills_df, university_skills):
    """
    Рассчитываем семантическую близость между набором навыков рынка труда и университета.

    Параметры:
        market_skills_df: DataFrame с навыками рынка труда
        university_skills: список навыков университета

    Результат выполнения:
        similarity_score: оценка семантической близости
    """

    all_skills = list(market_skills_df['skill']) + university_skills

    # создаем векторизатор и преобразуем навыки в матрицу частот
    vectorizer = CountVectorizer()
    skill_vectors = vectorizer.fit_transform(all_skills)

    # разделяем векторы на рыночные и университетские
    market_vectors = skill_vectors[:len(market_skills_df)]
    uni_vectors = skill_vectors[len(market_skills_df):]


    similarity_matrix = cosine_similarity(market_vectors.mean(axis=0), uni_vectors.mean(axis=0))
    similarity_score = similarity_matrix[0][0]

    return similarity_score


def identify_most_valuable_missing_skills(missing_skills_df, top_n=10):
    """
    Определяем наиболее ценные отсутствующие навыки, которые университет должен добавить.

    Параметры::
        missing_skills_df: DataFrame с отсутствующими навыками
        top_n: количество навыков для вывода

    Результат выполнения:
        top_missing_skills: DataFrame с топом отсутствующх навыков
    """
    return missing_skills_df.head(top_n)

#
def visualize_results(coverage, missing_skills_df, overlap_skills_df, market_skills_df):
    """
    

    Параметры:
        coverage: процент покрытия навыков
        missing_skills_df: DataFrame с отсутствующими навыками
        overlap_skills_df: DataFrame с перекрывающимися навыками
        market_skills_df: DataFrame с навыками рынка труда
    """
    plt.style.use('ggplot')

    weight_coverage = 0.5
    weight_similarity = 0.5
    adjusted_coverage = coverage * weight_coverage + similarity_score * 100 * weight_similarity


    # График 1: Круговая диаграмма покрытия навыков
    fig, ax = plt.subplots(figsize=(8, 8))
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    fig.suptitle('Общее соответствие программы требованиям рынка\n(с учётом покрытия и семантики)',
                 fontsize=16, weight='bold', color='#2c3e50')

    values = [adjusted_coverage, 100 - adjusted_coverage]
    labels = [f'Соответствие\n({adjusted_coverage:.1f}%)',
              f'Несоответствие\n({100 - adjusted_coverage:.1f}%)']

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda p: f'{p:.1f}%',
        startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 12, 'color': 'white'},
        colors=['#27ae60', '#c0392b'],
        explode=(0.04, 0)
    )

    ax.set_title(f'Coverage: {coverage:.1f}%  |  Semantic Similarity: {similarity_score * 100:.1f}%', fontsize=12)

    plt.savefig('skill_coverage_pie.png', dpi=300)
    plt.close()

    # График 2: Топ 10 отсутствующих навыков
    fig, ax = plt.subplots(figsize=(14, 8))
    top_missing = missing_skills_df.head(10)
    sns.barplot(x='normalized_weight', y='skill', data=top_missing, ax=ax, palette='Reds_r')
    ax.set_title('Топ-10 отсутствующих навыков', fontsize=18, fontweight='bold')
    ax.set_xlabel('Нормализованный вес', fontsize=14)
    ax.set_ylabel('Навык', fontsize=14)
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.2f}', (p.get_width() + 0.5, p.get_y() + p.get_height() / 2),
                    fontsize=12, color='black', va='center')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig('top_missing_skills.png', dpi=300)
    plt.close()

    #  График 3: Топ 10 перекрывающихся навыков
    fig, ax = plt.subplots(figsize=(14, 8))
    top_overlap = overlap_skills_df.head(10)
    sns.barplot(x='normalized_weight', y='skill', data=top_overlap, ax=ax, palette='Greens_r')
    ax.set_title('Топ-10 навыков, присутствующих в университете', fontsize=18, fontweight='bold')
    ax.set_xlabel('Нормализованный вес', fontsize=14)
    ax.set_ylabel('Навык', fontsize=14)
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.2f}', (p.get_width() + 0.5, p.get_y() + p.get_height() / 2),
                    fontsize=12, color='black', va='center')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig('top_overlap_skills.png', dpi=300)
    plt.close()

    # График 4: Распределение весов навыков на рынке труда
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.histplot(market_skills_df['normalized_weight'], bins=25, color='#3498db', kde=True, ax=ax)
    ax.set_title('Распределение весов навыков на рынке труда', fontsize=18, fontweight='bold')
    ax.set_xlabel('Нормализованный вес', fontsize=14)
    ax.set_ylabel('Количество навыков', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('market_skill_distribution.png', dpi=300)
    plt.close()

    #  График 5: Boxplot для оценки разброса весов навыков
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(y=market_skills_df['normalized_weight'], palette='autumn', ax=ax)
    ax.set_title('Boxplot распределения весов навыков', fontsize=18, fontweight='bold')
    ax.set_ylabel('Нормализованный вес', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('skill_weight_boxplot.png', dpi=300)
    plt.close()

    #  График 6: Распределение отсутствующих навыков
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.histplot(missing_skills_df['normalized_weight'], bins=25, color='#e67e22', kde=True, ax=ax)
    ax.set_title('Распределение отсутствующих навыков', fontsize=18, fontweight='bold')
    ax.set_xlabel('Нормализованный вес', fontsize=14)
    ax.set_ylabel('Количество навыков', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('missing_skills_distribution.png', dpi=300)
    plt.close()

    print("Графики успешно сохранены в файлы.")


def generate_report(coverage, missing_skills_df, overlap_skills_df, similarity_score):
    """
    Генерируем текстовый отчет с результатами анализа.

    Параметры:
        coverage: процент покрытия навыков
        missing_skills_df: DataFrame с отсутствующими навыками
        overlap_skills_df: DataFrame с перекрывающимися навыками
        similarity_score: оценка семантической близости

    Результат выполнения:
        report: строка с отчетом
    """
    report = f"""
ОТЧЕТ ПО АНАЛИЗУ РАЗРЫВА НАВЫКОВ
================================

Общая статистика:
----------------
- Всего навыков на рынке труда: {len(missing_skills_df) + len(overlap_skills_df)}
- Навыков в университетской программе: {len(overlap_skills_df)}
- Отсутствующих навыков: {len(missing_skills_df)}
- Взвешенное покрытие навыков: {coverage:.2f}%
- Семантическая близость программ: {similarity_score:.2f} (от 0 до 1)

Топ-10 наиболее востребованных навыков, отсутствующих в университете:
-------------------------------------------------------------------
"""

    for i, (_, row) in enumerate(missing_skills_df.head(10).iterrows()):
        report += f"{i + 1}. {row['skill']} (вес: {row['normalized_weight']:.2f})\n"

    report += """
Топ-10 наиболее востребованных навыков, присутствующих в университете:
-------------------------------------------------------------------
"""

    for i, (_, row) in enumerate(overlap_skills_df.head(10).iterrows()):
        report += f"{i + 1}. {row['skill']} (вес: {row['normalized_weight']:.2f})\n"

    return report


def main(market_skills_path, university_skills_path, output_report_path='skill_gap_report_SYS_admin.txt'):
    """
    Основная функция

    Параметры:
        market_skills_path: путь к файлу с навыками рынка труда
        university_skills_path: путь к файлу с навыками университета
        output_report_path: путь для сохранения отчета
    """

    market_skills_df, university_skills = load_and_preprocess_data(
        market_skills_path, university_skills_path
    )


    coverage, missing_skills_df, overlap_skills_df = analyze_skill_gap(
        market_skills_df, university_skills
    )

    # семантическуя близость
    similarity_score = calculate_similarity(market_skills_df, university_skills)

    # визуализируем результаты
    visualize_results(coverage, missing_skills_df, overlap_skills_df, market_skills_df)

    # генерируем отчет
    report = generate_report(coverage, missing_skills_df, overlap_skills_df, similarity_score)
    with open(output_report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Анализ завершен. Отчет сохранен в {output_report_path}")
    print(f"Визуализации сохранены в skill_gap_analysis_test.png")

    return coverage, missing_skills_df, overlap_skills_df, similarity_score


if __name__ == "__main__":

    market_skills_path = "market.csv"


    university_skills_path = "university.csv"




    main(market_skills_path, university_skills_path)
