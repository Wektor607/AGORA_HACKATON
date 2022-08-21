# DjangoAPI SOLUTION

# Общее решение

1. Скачать DjangoAPI.zip, распаковать;
2. Перейти в директорию `\DjangoAPI`;
3. Запуск сервера через команду `python manage.py runserver`;
4. Теперь запросы по указанному сформировать json-файл для отправки на сервер
    - Использовать HTTP-запрос `POST http://127.0.0.1:8000/match_products`;
    - Получить в теле ответа json с предсказаниями;

# Обучение моделей

1. Скачать **TwoModels.py**, **arcNet.py**, **Ensemble.py**, в этой же директории `agora_hack_products.json`
2. ExtraTreesClassifier[Классическое ML-решение] и Tokenizator[Использование данных и их сопоставление]:
    1. Запустить команду `python TwoModels.py train` -- в результате сохранятся модели и дополнительные файлы;
    2. Запуск команды `python TwoModels.py token` покажет результат работы Tokenizator'а на всех данных;
    3. Запуск команды `python TwoModels.py test` покажет результат работы ExtraTreeClassifier'а на всех данных;
3. ArcNet[Нейросетевое решение]:
    1. Запуск команды `python arcNet.py` выведет ожидаемые параметры на вход py-файлу;
    2. Запуск команды `python arcNet.py train [true/false] [true/false] [true/false] [true/false] [true/false]` обучит модели и по требованию сохранит их в файлы, так же сохранят дополнительные файлы;
    3. Запуск команды `python arcNet.py test [true/false]` проверит работу модели на тестовых данных, так же сохранит при необходимости голову модели(KNN-1)
    4. Запуск команды `python arcNet.py check` покажет точность и время работы на всех данных;
