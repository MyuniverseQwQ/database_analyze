# main.py
from modules.preprocesser import DataPreprocessor
from modules.visualizer import Visualizer


def main():
    # 1. Путь к исходному датасету
    # Укажите ваш путь, если файл в другом месте
    raw_data_path = "database/credit_risk_dataset.csv"

    # 2. Запускаем предобработку
    preprocessor = DataPreprocessor(filePath=raw_data_path)
    preprocessor.run_full_pipeline()

    # 3. Запускаем визуализацию
    processed_data_path = "dataset_processed/credit_risk_dataset_processed.csv"

    # Создаем объект класса Visualizer и запускаем графики
    viz = Visualizer(processed_data_path)
    viz.run_visualization()


if __name__ == "__main__":
    main()
