import pandas as pd
import os


class DataPreprocessor:
    """
    Класс для загрузки и предобработки данных кредитного риска
    """

    def __init__(self, filePath=None, outputPath=None):
        """
        Конструктор класса
        параметры:
        -----------
        filePath : str, optional
            Путь к файлу с данными
        outputPath : str, optional
            Путь для сохранения обработанных данных
        """
        self.filePath = filePath
        default_csv = 'dataset_processed/credit_risk_dataset_processed.csv'
        self.output_path = outputPath or default_csv
        self.df = None
        self.initial_shape = None
        # Категориальные колонки для анализа
        self.categorical_cols = [
            'person_home_ownership', 'loan_intent',
            'loan_grade', 'cb_person_default_on_file'
        ]

    def load_data(self, filePath=None):
        """
        Загрузка данных из CSV файла
        параметры:
        -----------
        filePath : str, optional
            Путь к файлу (если не указан при инициализации)
        """
        if filePath:
            self.filePath = filePath
        elif not self.filePath:
            self.filePath = input(
                "Введите полный путь к файлу датасета: "
            )
        print(f"Загрузка данных из: {self.filePath}")
        self.df = pd.read_csv(fr'{self.filePath}')
        self.initial_shape = self.df.shape
        print(f"Данные загружены. Размер: {self.initial_shape}")
        return self

    def analyze_data(self):
        """
        Анализ исходного датасета
        """
        print("\n" + "=" * 60)
        print("1. АНАЛИЗ ИСХОДНОГО ДАТАСЕТА")
        print("=" * 60)
        # Размерность
        print(f"\nРАЗМЕР ДАТАСЕТА: {self.df.shape}")
        # Типы переменных
        print(f"\nТИПЫ ДАННЫХ:\n{self.df.dtypes}")
        # Примеры данных
        print("\nПРИМЕРЫ ДАННЫХ (первые 5 строк):")
        print(self.df.head())
        # Пропуски
        print(f"\nКОЛИЧЕСТВО ПРОПУСКОВ:\n{self.df.isnull().sum()}")
        # Статистика числовых
        print(f"\nСТАТИСТИКА ЧИСЛОВЫХ СТОЛБЦОВ:\n{self.df.describe()}")
        # Статистика категориальных
        print("\nСТАТИСТИКА КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ:")
        for col in self.categorical_cols:
            if col in self.df.columns:
                print(f"\n{col}:")
                print(f"  Уникальных: {self.df[col].nunique()}")
                print(f"  Частоты:\n{self.df[col].value_counts()}")
        # Выявленные проблемы
        print("\nВЫЯВЛЕННЫЕ ПРОБЛЕМЫ:")
        print(
            "1. Пропуски: loan_int_rate (450 шт., 1.38%), "
            "person_emp_length (896 шт., 2.75%)"
        )
        print(
            "2. Выбросы: person_age = 144 года, "
            "person_emp_length = 123 года"
        )
        print(
            "3. Несогласованность форматов: "
            "cb_person_default_on_file (Y/N), loan_grade (A-G)"
        )
        return self

    def remove_outliers(self):
        """
        Удаление выбросов из данных
        """
        print("\n--- УДАЛЕНИЕ ВЫБРОСОВ ---")
        rows_before = len(self.df)
        self.df = self.df[self.df["person_age"] <= 100]
        self.df = self.df[
            self.df["person_emp_length"]
            <= (self.df["person_age"] - 16)
        ]
        self.df = self.df[
            (self.df["person_income"] > 0) & (self.df["loan_amnt"] > 0)
        ]

        rows_removed = rows_before - len(self.df)
        print(f"Удалено строк: {rows_removed}")

        return self

    def fill_missing_values(self):
        """
        Заполнение пропущенных значений
        """
        print("\n--- ЗАПОЛНЕНИЕ ПРОПУСКОВ ---")

        # Заполнение медианой
        self.df['person_emp_length'] = self.df[
            'person_emp_length'
        ].fillna(self.df['person_emp_length'].median())

        # Заполнение по группам loan_grade
        self.df['loan_int_rate'] = self.df.groupby('loan_grade')[
            'loan_int_rate'
        ].transform(lambda x: x.fillna(x.median()))
        self.df['loan_int_rate'] = self.df['loan_int_rate'].fillna(
            self.df['loan_int_rate'].median()
        )

        print(f"Пропусков после обработки: {self.df.isnull().sum().sum()}")

        return self

    def encode_categorical_features(self):
        """
        Кодирование категориальных признаков
        """
        print("\n--- КОДИРОВАНИЕ ПРИЗНАКОВ ---")

        # Бинарное кодирование
        self.df['cb_person_default_on_file'] = (
            self.df['cb_person_default_on_file']
            .map({'Y': 1, 'N': 0})
        )

        # Порядковое кодирование loan_grade
        grade_mapping = {
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7
        }
        self.df['loan_grade_encoded'] = (
            self.df['loan_grade'].map(grade_mapping)
        )

        # One-hot encoding
        self.df = pd.get_dummies(
            self.df,
            columns=['person_home_ownership'],
            prefix='home',
            drop_first=True
        )
        self.df = pd.get_dummies(
            self.df,
            columns=['loan_intent'],
            prefix='intent',
            drop_first=True
        )

        # Удаление исходного столбца
        self.df = self.df.drop(columns=['loan_grade'])

        return self

    def create_features(self):
        """
        Создание новых признаков
        """
        print("\n--- СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ ---")

        self.df['cred_hist_to_age_ratio'] = (
            self.df['cb_person_cred_hist_length'] / self.df['person_age']
        )
        self.df['income_per_emp_year'] = (
            self.df['person_income'] / (self.df['person_emp_length'] + 1)
        )

        return self

    def get_results(self):
        """
        Вывод результатов предобработки
        """
        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТ ПРЕДОБРАБОТКИ")
        print("=" * 50)

        print(f"Финальный размер: {self.df.shape}")
        print(f"Пропуски: {self.df.isnull().sum().sum()}")
        print("\nСРАВНЕНИЕ С ИСХОДНЫМ ДАТАСЕТОМ:")
        print(
            f"  Исходный: {self.initial_shape[0]} строк, "
            f"{self.initial_shape[1]} столбцов"
        )
        print(
            f"  После обработки: {self.df.shape[0]} строк, "
            f"{self.df.shape[1]} столбцов"
        )

        rows_removed = self.initial_shape[0] - self.df.shape[0]
        removal_percent = (rows_removed / self.initial_shape[0]) * 100
        print(
            f"  Удалено строк: {rows_removed} ({removal_percent:.2f}%)"
        )

        return self

    def save_data(self, outputPath=None):
        """
        Сохранение обработанных данных

        параметры:
        -----------
        outputPath : str, optional
            Путь для сохранения
        """
        if outputPath:
            self.output_path = outputPath

        # Создание директории, если её нет
        dir_name = os.path.dirname(self.output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        self.df.to_csv(self.output_path, index=False)
        print(f"\nОбработанный датасет сохранён: {self.output_path}")

        return self

    def preprocess(self, save=True, outputPath=None):
        """
        Выполнение полного цикла предобработки

        параметры:
        -----------
        save : bool
            Сохранять ли результат
        outputPath : str, optional
            Путь для сохранения
        """
        print("\n" + "=" * 60)
        print("НАЧАЛО ПРЕДОБРАБОТКИ ДАННЫХ")
        print("=" * 60)

        self.remove_outliers()
        self.fill_missing_values()
        self.encode_categorical_features()
        self.create_features()
        self.get_results()

        if save:
            self.save_data(outputPath)

        print("\n" + "=" * 60)
        print("ПРЕДОБРАБОТКА ЗАВЕРШЕНА")
        print("=" * 60)

        return self.df

    def run_full_pipeline(self, filePath=None, outputPath=None):
        """
        Запуск полного пайплайна: загрузка -> анализ -> предобработка

        параметры:
        -----------
        filePath : str, optional
            Путь к файлу с данными
        outputPath : str, optional
            Путь для сохранения результата
        """
        self.load_data(filePath)
        self.analyze_data()
        return self.preprocess(save=True, outputPath=outputPath)
