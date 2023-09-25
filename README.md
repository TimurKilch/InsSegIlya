# Описание репозитория
В данном репозитории представлен программный код формирования абрисов объектов на спутниковом снимке местности.

Для использования данного ПО в сборках Docker Compose создан образ Docker на основе сервера Nginx с поддержкой протокола обмена данными FastCGI.

## Состав репозитория
Репозиторий включает в себя две части: исходный код (папка `sources`) и реализацию данного кода в виде образа Docker (папка `docker`).

### Содержимое папки sources
Включает в себя три файла:

1) `main.py` - исходный код формирования абрисов целей на языке Python;
2) `requirements.system` - перечень пакетов apt, необходимых для корректной работы исходного кода;
3) `requirements.txt` - перечень пакетов pip, необходимых для корректной работы исходного кода.

### Содержимое папки docker
Состав папки представлен так же тремя файлами:

1) `default.conf` - конфигурационный файл сервера Nginx (лежащего в основе Docker образа);
2) `docker-entrypoint.sh` - bash скрипт, является входной точкой при запуске Docker контейнера, выполняет загрузку самой последней версии исходного кода из данного репозитория;
3) `fcgiwrap.conf` - конфигурационный файл Nginx, содержащий путь до исходного кода внутри контейнера.

Для корректной работы системы CI/CD основной Dockerfile образа вынесен в корень данного репозитория.

## Использование ПО

В качестве входных данных выступает спутниковое изображение, удовлетворяющее [требованиям](#требования-к-входным-данным) настоящего ПО. По завершении работы ПО формируется так называемый "словарь" формата `.json` или `.geojson`. 

Совокупность спутникового снимка и "словаря" позволяет получить изображение с обозначенными на нём абрисами (границами) объектов (зданий, сооружений, территорий).

### Вызов ПО из терминала
Выполняется командой с указанием двух аргументов:

`
python3 /путь_до_файла/main.py /путь_до_входного_изображения /желаемый_путь_словаря
`
### Требования к входным данным
Изображение, выступающее в качестве входных данных, должно удовлетворять следующим требованиям:

1) Расширение `.tif`, `.tiff`, `.geotiff`, `.png`, `.jpeg`, `.jpg`;
2) Точность изображения 1 м/пиксель (допустимо незначительное отклонение +- 0,2 м/пиксель).

С увеличением объёма изображения возможно снижение скорости работы ПО.