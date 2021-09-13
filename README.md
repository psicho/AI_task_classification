# Валидация задач по категориям Good/Agood/Bad

## Продукт предназначен для классификации задач на Good/Agood/Bad, а также определения группы задачи.

### 1. Валидация задач на три группы (столбец "Good/Agood/Bad")
    - good (хорошие)
    - bad (пложие) 
    - agood (скорее хорошие)


### 2. Определение группы задачи (столбец "Group")
    - Multiplication and division
    - Addition and subtraction
    - Fractions
    - Mixed operations
    - Measurements
    - Figures
    - Number
    - Modelling
    - Geometry
    - Time
    - Comparison
    - Estimation
    - Logic
    - Series and pattern
    - Graph
    - Probability
    - Money
    - Other

### Установка зависимостей и запуск проекта
    Для работы проекта необходимо установить следующие зависимости:
    pip install spacy
    pip install pandas
    pip install jupyter
    pip install anaconda
    pip install sklearn
    spacy download en_core_web_sm

    Загрузить файл обученной модели и добавить его в раздел "training_models" проекта 
    https://drive.google.com/drive/folders/1bh_TLIvu9ot-97Bg--tcWemx4hixaV8w
    (размер файла модели более 400 Мб и github не поддерживает загрузку файлов более 100 Мб)

    
## Быстрый старт
1. Запустите Jupyter Notebook
    -  выполните команду в терминале: **jupyter notebook**
2. В поле **url** добавьте ссылку на файл с задачами в **docs.google.com**
3. Запустите выполнение модуля **token.classification_data_set(url)**
4. После окончания работы модуля итоговый файл можно найти в директории проекта: **Data**
