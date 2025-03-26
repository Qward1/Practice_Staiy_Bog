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


def visualize_results(coverage, missing_skills_df, overlap_skills_df, market_skills_df):
    """
    Улучшенная визуализация результатов анализа навыков с современным дизайном.
    """
    # Настройка стиля и параметров
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("viridis")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 12

    # Создание комплексной фигуры
    fig = plt.figure(figsize=(22, 18), dpi=300)
    gs = fig.add_gridspec(3, 3)
    fig.suptitle('Комплексный анализ соответствия учебных программ требованиям рынка труда',
                 fontsize=20, y=1.02, weight='bold', color='#2c3e50')

    # График 1: Круговая диаграмма с интерактивными элементами
    ax1 = fig.add_subplot(gs[0, 0])
    explode = (0.05, 0.05)
    wedges, texts, autotexts = ax1.pie(
        [coverage, 100 - coverage],
        labels=['Покрытие', 'Дефицит'],
        autopct=lambda p: f'{p:.1f}%\n({int(p / 100 * len(market_skills_df))} навыков)',
        colors=['#27ae60', '#c0392b'],
        startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 3, 'linestyle': '--'},
        textprops={'fontsize': 11, 'color': 'white'},
        explode=explode,
        shadow=True
    )
    ax1.set_title('Соответствие учебной программы рынку труда\n', fontsize=14)

    # График 2: Топ отсутствующих навыков (вертикальный)
    ax2 = fig.add_subplot(gs[0, 1:])
    top_missing = missing_skills_df.head(10).sort_values('normalized_weight', ascending=False)
    bars = sns.barplot(x='skill', y='normalized_weight', data=top_missing, ax=ax2,
                       palette='rocket_r', edgecolor='black')
    ax2.set_title('Топ-10 критических недостающих навыков', fontsize=14)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_xlabel('')
    ax2.set_ylabel('Важность на рынке', fontsize=12)

    # Добавление значений на столбцы
    for i, (_, row) in enumerate(top_missing.iterrows()):
        ax2.text(i, row.normalized_weight + 0.01, f'{row.normalized_weight:.2f}',
                 ha='center', fontsize=10, rotation=45)

    # График 3: Сравнение распределений
    ax3 = fig.add_subplot(gs[1, 0])
    sns.violinplot(data=[market_skills_df['normalized_weight'],
                         missing_skills_df['normalized_weight']],
                   palette=['#2980b9', '#e67e22'], ax=ax3)
    ax3.set_title('Сравнение распределения важности навыков', fontsize=14)
    ax3.set_xticklabels(['Рыночные навыки', 'Дефицитные навыки'])
    ax3.set_ylabel('Нормализованная важность', fontsize=12)

    # График 4: Тепловая карта топ-20 навыков
    ax4 = fig.add_subplot(gs[1, 1:])
    combined_skills = pd.concat([
        overlap_skills_df.head(20),
        missing_skills_df.head(20)
    ]).drop_duplicates().head(20)

    pivot_data = combined_skills.pivot_table(index='skill',
                                             values='normalized_weight',
                                             aggfunc='mean').sort_values(
        'normalized_weight', ascending=False)
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=.5, ax=ax4, cbar_kws={'label': 'Уровень важности'})
    ax4.set_title('Тепловая карта ключевых навыков', fontsize=14)
    ax4.set_xlabel('')
    ax4.set_ylabel('')
    plt.setp(ax4.get_yticklabels(), rotation=0)

    # График 5: Комбинированный график распределения
    ax5 = fig.add_subplot(gs[2, :2])
    sns.histplot(market_skills_df['normalized_weight'], bins=30, kde=True,
                 element='step', alpha=0.6, label='Общий рынок', ax=ax5)
    sns.histplot(missing_skills_df['normalized_weight'], bins=30, kde=True,
                 element='step', alpha=0.6, color='#e74c3c', label='Дефицит', ax=ax5)
    ax5.set_title('Сравнительное распределение важности навыков', fontsize=14)
    ax5.set_xlabel('Нормализованная важность', fontsize=12)
    ax5.set_ylabel('Плотность', fontsize=12)
    ax5.legend()




    # Общие настройки
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)

    # Сохранение и вывод

    filename = f'skills_analysis_CyberSecurity.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Комплексный отчет сохранен как: {filename}")

def visualize_results(coverage, missing_skills_df, overlap_skills_df, market_skills_df):
    """
    Улучшенная визуализация с исправлением ошибок и дополнительной проверкой данных
    """
    # Проверка наличия необходимых данных
    if any([df.empty for df in [market_skills_df, missing_skills_df, overlap_skills_df]]):
        raise ValueError("Один из DataFrame пуст. Проверьте входные данные")

    # Настройка стиля
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # Создание комплексной фигуры
    fig = plt.figure(figsize=(24, 20), dpi=300)
    gs = fig.add_gridspec(3, 3)
    fig.suptitle('Комплексный анализ соответствия учебных программ требованиям рынка труда\n',
                 fontsize=18, y=1.02, weight='bold', color='#2c3e50')

    # График 1: Круговая диаграмма покрытия
    ax1 = fig.add_subplot(gs[0, 0])
    coverage_values = [coverage, 100 - coverage]
    labels = [f'Покрытие ({len(overlap_skills_df)} навыков)',
              f'Дефицит ({len(missing_skills_df)} навыков)']

    wedges, texts, autotexts = ax1.pie(
        coverage_values,
        labels=labels,
        autopct=lambda p: f'{p:.1f}%',
        startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 10, 'color': 'white'},
        colors=['#2ecc71', '#e74c3c'],
        explode=(0.03, 0)
    )
    ax1.set_title(f'Покрытие навыков рынка труда: {coverage:.1f}%\n', fontsize=14)

    # График 2: Топ отсутствующих навыков
    ax2 = fig.add_subplot(gs[0, 1:])
    top_missing = missing_skills_df.head(10)
    bars = sns.barplot(
        x='normalized_weight',
        y='skill',
        data=top_missing,
        ax=ax2,
        palette='Reds_r',
        edgecolor='black'
    )

    # Аннотации для барплота

    ax2.set_title(f'Топ-10 критических недостающих навыков\n',
                  fontsize=14)
    ax2.set_xlabel('Нормализованный вес (рыночная значимость)', fontsize=12)
    ax2.set_ylabel('')
    ax2.xaxis.grid(True, linestyle='--', alpha=0.6)

    # График 3: Исправленный violinplot
    ax3 = fig.add_subplot(gs[1, 0])
    market_data = market_skills_df[['normalized_weight']].assign(category='Рыночные навыки')
    missing_data = missing_skills_df[['normalized_weight']].assign(category='Дефицитные навыки')
    combined_dist = pd.concat([market_data, missing_data], axis=0)

    sns.violinplot(
        x='category',
        y='normalized_weight',
        data=combined_dist,
        palette=['#2980b9', '#e67e22'],
        ax=ax3
    )

    # Аннотации статистик
    stats_text = (
        f"Рыночные навыки:\n"
        f"Медиана: {market_skills_df['normalized_weight'].median():.2f}\n"
        f"Среднее: {market_skills_df['normalized_weight'].mean():.2f}\n\n"
        f"Дефицитные навыки:\n"
        f"Медиана: {missing_skills_df['normalized_weight'].median():.2f}\n"
        f"Среднее: {missing_skills_df['normalized_weight'].mean():.2f}"
    )
    ax3.text(
        0.95, 0.95,
        stats_text,
        transform=ax3.transAxes,
        ha='right',
        va='top',
        fontsize=9,
        bbox=dict(facecolor='white', alpha=0.8)
    )

    ax3.set_title('Сравнение распределения важности навыков', fontsize=14)
    ax3.set_ylabel('Нормализованный вес', fontsize=12)

    # График 4: Тепловая карта с проверкой данных
    ax4 = fig.add_subplot(gs[1, 1:])
    combined_skills = pd.concat([
        overlap_skills_df.head(20),
        missing_skills_df.head(20)
    ])
    combined_skills = combined_skills.drop_duplicates(subset=['skill']).head(20)

    if not combined_skills.empty:
        pivot_table = combined_skills.pivot_table(
            index='skill',
            values='normalized_weight',
            aggfunc='mean'
        ).sort_values('normalized_weight', ascending=False)

        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            linewidths=0.5,
            ax=ax4,
            cbar_kws={'label': 'Уровень важности'}
        )
    else:
        ax4.text(0.5, 0.5, 'Нет данных для визуализации', ha='center', va='center')

    ax4.set_title('Тепловая карта ключевых навыков (Топ-15 совпадающих и недостающих)', fontsize=14)
    ax4.xaxis.set_visible(False)
    plt.setp(ax4.get_yticklabels(), rotation=0, fontsize=10)

    # График 5: Комбинированное распределение с KDE
    ax5 = fig.add_subplot(gs[2, :2])
    sns.histplot(
        market_skills_df['normalized_weight'],
        bins=30,
        kde=True,
        color='#3498db',
        label='Все рыночные навыки',
        alpha=0.5,
        ax=ax5
    )
    sns.histplot(
        missing_skills_df['normalized_weight'],
        bins=30,
        kde=True,
        color='#e74c3c',
        label='Дефицитные навыки',
        alpha=0.5,
        ax=ax5
    )

    # Вертикальные линии для средних значений
    ax5.axvline(
        market_skills_df['normalized_weight'].mean(),
        color='#3498db',
        linestyle='--',
        linewidth=1.5,
        label='Среднее для рынка'
    )
    ax5.axvline(
        missing_skills_df['normalized_weight'].mean(),
        color='#e74c3c',
        linestyle='--',
        linewidth=1.5,
        label='Среднее для дефицита'
    )

    ax5.set_title('Сравнение распределений важности навыков', fontsize=14)
    ax5.set_xlabel('Нормализованный вес', fontsize=12)
    ax5.set_ylabel('Плотность', fontsize=12)
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.6)

    # График 6: Анализ покрытия по квантилям
    ax6 = fig.add_subplot(gs[2, 2])
    quantiles = market_skills_df['normalized_weight'].quantile([0.25, 0.5, 0.75, 0.9])
    coverage_by_quantile = []

    for q in quantiles:
        covered = overlap_skills_df[overlap_skills_df['normalized_weight'] >= q].shape[0]
        total = market_skills_df[market_skills_df['normalized_weight'] >= q].shape[0]
        coverage_by_quantile.append((covered / total) * 100 if total > 0 else 0)

    sns.barplot(
        x=[f'Q{int(q * 100)}%' for q in [0.25, 0.5, 0.75, 0.9]],
        y=coverage_by_quantile,
        palette='viridis',
        ax=ax6
    )

    for i, v in enumerate(coverage_by_quantile):
        ax6.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)

    ax6.set_title('Покрытие по уровням важности', fontsize=14)
    ax6.set_ylabel('Процент покрытия', fontsize=12)
    ax6.set_ylim(0, 100)
    ax6.yaxis.grid(True, linestyle='--', alpha=0.6)

    # Общие настройки
    plt.tight_layout()
    fig.subplots_adjust(top=0.93, hspace=0.35, wspace=0.25)

    # Сохранение
    filename = 'sskill_gap_analysis.png'
    plt.savefig(filename, bbox_inches='tight', dpi=400)
    plt.close()

    print(f"Расширенный отчет сохранен как: {filename}")



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