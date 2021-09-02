# Валидация задач по категориям Good/Agood/Bad

## Продукт предназначен для классификации задач на Good/Agood/Bad, а также определения группы задачи.

### 1. Валидация задач на три группы
    - good (хорошие)
    - bad (пложие) 
    - agood (скорее хорошие)


### 2. Определение группы задачи
    - number_properties
    - geometry
    - measurement
    - algebra
    - data_and_probability

### Установка зависимостей и запуск проекта
    Для работы проекта необходимо установить следующие зависимости:
    pip install spacy
    pip install pandas
    pip install jupyter
    pip install anaconda
    pip install sklearn
    spacy download en_core_web_sm
    
## Быстрый старт
1. Запустите Jupyter Notebook
    -  выполните команду в терминале: **jupyter notebook**
2. В поле **url** добавьте ссылку на файл с задачами в **docs.google.com**
3. Запустите выполнение модуля **token.classification_data_set(url)**
4. После окончания работы модуля итоговый файл можно найти в директории проекта: **Data**
