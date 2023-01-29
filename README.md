# Style-transfer-Bot
Telegram bot for style transfer

Привет!

Этот бот создан в рамках учебного проекта по первой части курса DLS ([Deep learning school](https://dls.samcs.ru/)), и он умеет накладывать стиль одной картинки на другую.

Исходная картинка:

Принцип работы нейросети взят из [руководства](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) Pytorch. В основе - натренированная сеть VGG19, в которую добавлены дополнительные модули нормализации и лоссов.

Сам бот написан с помощью асинхронной библиотеки aiogram, обновления собираются с помощью поллинга.

Бот работает на удаленном сервере [Selectel](https://selectel.ru/). Деплой выполнен по руководству из [курса](https://stepik.org/course/120924/) "Telegram-боты на Python и AIOgram" на платформе Stepik (часть 11.1 - Переселение бота на сервер).
Конфигурация сервера:
* Ubuntu 22.04 LTS 64-bit
* Standard Line, 2 ядра, 10%, 4 ГБ
* диск HDD Базовый, 10 Гб
(возможно, не самая оптимальная, достигнута авторскими методами тыка и проб и ошибок). Если, дорогой читатель, ты решишь повторить процесс и взять этот репозиторий, скорее всего, тебе потребуется парочка дополнительных pip install (например, в моем случае aiogram, environs и torch - версию последнего можно выбрать тут).

Демонстрацию работы бота и скрины статистики сервера можно найти [тут](https://drive.google.com/drive/folders/18Q2qhJqBOhglSzdTptGrBD5JBoWwocP2?usp=sharing
