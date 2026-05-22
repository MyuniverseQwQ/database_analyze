import pandas as pd       # Библиотека для работы с таблицами (DataFrame)
import matplotlib.pyplot as plt  # Основная библиотека для построения графиков
# Библиотека для работы с файловой системой (создание папок)
import os


class Visualizer:
    """
    Класс для визуализации предобработанных данных.
    Отвечает за построение, оформление и сохранение графиков.
    """

    def __init__(self, filepath, output_dir="plots"):
        """
        Конструктор класса. Вызывается при создании объекта Visualizer.

        параметры:
        -----------
        filepath : str
            Путь к очищенному CSV файлу, который выдаст DataPreprocessor
        output_dir : str
            Папка, куда будут сохраняться картинки с графиками
        """
        self.filepath = filepath
        self.output_dir = output_dir

        # Проверяем, существует ли папка для графиков. Если нет — создаем её
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Сразу загружаем данные из CSV в таблицу pandas
        print(f"Загрузка данных для визуализации из: {self.filepath}")
        self.df = pd.read_csv(self.filepath)
        print(f"Данные загружены. Строк: {len(self.df)}")

    def _save_and_show(self, fig, filename):
        """
        Отвечает за сохранение картинки в файл и её отображение на экране.
        Вызывается в конце каждого метода отрисовки графика.
        """
        # Склеиваем имя папки и имя файла в один путь
        filepath = os.path.join(self.output_dir, filename)

        # Сохраняем график. dpi=300 — высокое разрешение для отчета.
        # bbox_inches='tight' — обрезает пустые белые поля по краям графика
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {filepath}")

        # plt.show() открывает окно с графиком на экране
        plt.show()

        # plt.close() обязательно закрывает график, чтобы он не висел в
        # оперативной памяти
        plt.close(fig)

    def plot_age_vs_loan_status(self):
        """
        График 1: Распределение возраста по платежной дисциплине.
        Показывает, в каком возрасте люди чаще возвращают кредит,
        а в каком допускают просрочки.
        """
        # Создаем холст (fig) и оси координат (ax). Размер задаем в дюймах (10
        # на 6)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Фильтруем датафрейм: берем только надежных (статус 0) и только
        # столбец возраста
        reliable = self.df[self.df['loan_status'] == 0]['person_age']
        # Фильтруем датафрейм: берем проблемных (статус 1)
        risky = self.df[self.df['loan_status'] == 1]['person_age']

        # Строим гистограмму для надежных. bins=30 - разбиваем на 30 столбиков.
        # alpha=0.6 - полупрозрачность, чтобы столбики не перекрывали друг
        # друга наглухо
        ax.hist(
            reliable,
            bins=30,
            alpha=0.6,
            label='Надежный заемщик (0)',
            color='blue')
        # Строим гистограмму для проблемных на тех же осях
        ax.hist(
            risky,
            bins=30,
            alpha=0.6,
            label='Проблемный заемщик (1)',
            color='red')

        # Подписываем заголовок и оси
        ax.set_title('Распределение возраста по платежной дисциплине')
        ax.set_xlabel('Возраст (лет)')
        ax.set_ylabel('Количество клиентов')

        # Добавляем пунктирную сетку для удобства чтения значений по Y
        ax.grid(True, linestyle='--', alpha=0.7)
        # Показываем легенду (расшифровка цветов)
        ax.legend()

        # Вызываем наш метод для сохранения картинки в файл
        self._save_and_show(fig, "1_age_vs_loan_status.png")

    def plot_loan_intent(self):
        """
        График 2: Распределение целей кредитования.
        Показывает, на какие цели люди берут кредиты чаще всего.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # 1. Находим все столбцы, которые начинаются на 'intent_'
        intent_cols = [
            col for col in self.df.columns if col.startswith('intent_')]

        # 2. Считаем количество единичек (1) в каждом таком столбце.
        # Сумма столбца из нулей и единиц даст общее количество таких кредитов
        counts = self.df[intent_cols].sum().sort_values(ascending=False)

        # 3. Очищаем названия для графика (убираем приставку 'intent_')
        # Например, 'intent_EDUCATION' превратится в 'EDUCATION'
        labels = [col.replace('intent_', '') for col in counts.index]

        # Строим столбчатую диаграмму по полученным данным
        ax.bar(labels, counts.values, color='skyblue', edgecolor='black')

        ax.set_title('Распределение целей кредитования')
        ax.set_xlabel('Цель кредита')
        ax.set_ylabel('Количество заявок')

        plt.xticks(rotation=30, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        self._save_and_show(fig, "2_loan_intent.png")

    def plot_loan_by_grade(self):
        """
        График 3: Сумма кредита по кредитному рейтингу.
        Показывает медианную сумму, разброс и выбросы для каждого рейтинга.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # 1. Получаем уникальные значения рейтингов, убираем пустые
        # и ОБЯЗАТЕЛЬНО приводим к целым числам (int), чтобы избежать бага 1.0,
        # чтобы избежать ошибок 1.0
        grades_encoded = sorted(
            self.df['loan_grade_encoded'].dropna().unique().astype(int))

        # 2. Создаем словарь для обратной расшифровки цифр в буквы
        grade_labels = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G'}

        # Формируем список букв-подписей в правильном порядке
        labels = [grade_labels.get(g, str(g)) for g in grades_encoded]

        # 3. Собираем данные для каждого числового рейтинга
        data_to_plot = [self.df[self.df['loan_grade_encoded']
                                == g]['loan_amnt'] for g in grades_encoded]

        medianprops = dict(linewidth=2, color='firebrick')

        # Строим botplox, передавая настройки
        ax.boxplot(
            data_to_plot,
            labels=labels,
            patch_artist=True,
            showfliers=False,
            medianprops=medianprops
        )

        explanation_text = (
            "Красная линия — Медиана (50% людей)\n"
            "Верх блока — 75% процентов\n"
            "Низ блока — 25% процентов\n"
            "Границы сверху и снизу — разброс без аномалий"
        )

        # Размещаем пояснительный внутри графика (в координатах от 0 до 1)
        # x=0.98, y=0.98 — правый верхний угол
        ax.text(
            0.98, 0.98,
            explanation_text,
            transform=ax.transAxes,  # Используем координаты области графика
            fontsize=10,
            verticalalignment='top',  # Выравнивание текста по верху
            horizontalalignment='right',  # Выравнивание текста по правому краю
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='gray',
                boxstyle='round,pad=0.5')  # Рамка
        )

        ax.set_title('Распределение суммы кредита по кредитному рейтингу')
        ax.set_xlabel('Кредитный рейтинг')
        ax.set_ylabel('Сумма кредита ($)')
        ax.grid(True, linestyle='--', alpha=0.7)

        self._save_and_show(fig, "3_loan_by_grade.png")

    def run_visualization(self):
        """
        Главный метод класса.
        Вызывает по очереди все методы отрисовки графиков.
        """
        print("\nНачало построения графиков...")
        self.plot_age_vs_loan_status()
        self.plot_loan_intent()
        self.plot_loan_by_grade()
        print("\nВсе графики успешно построены и сохранены в папку 'plots'!")
