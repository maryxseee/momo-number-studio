# Розпізнавання рукописних цифр: порівняння трьох моделей

**Автор: Колесник Марія · КН-3, група 1**

Навчальний проєкт із машинного навчання на MNIST. На однакових даних навчено LogisticRegression, MLP та CNN. Порівнюються accuracy, macro-F1, час навчання, швидкість прогнозування, кількість параметрів і розмір моделей. CNN з аугментацією — додатковий експеримент зі стійкістю, а не окрема архітектура.

## З чого почати перевірку

1. [Виконаний Notebook](mnist_project/research.ipynb) — пояснення, код, результати й графіки.
2. [Повний звіт](mnist_project/REPORT.md) — методика, результати, аналіз та обмеження.
3. [Порівняння трьох моделей](mnist_project/COMPARISON.md) — підсумкова таблиця якості й витрат.
4. [Відповідність вимогам](mnist_project/REQUIREMENTS.md) — де знайти кожний елемент роботи.

## Мапа репозиторію

### Дослідження та джерела

| Файл або папка | Призначення |
| --- | --- |
| [mnist_project/](mnist_project/) | Основна дослідницька робота |
| [README дослідження](mnist_project/README.md) | Встановлення, запуск і відтворення |
| [research.ipynb](mnist_project/research.ipynb) | Виконаний Notebook із кодом і результатами |
| [REPORT.md](mnist_project/REPORT.md) | Повний дослідницький звіт |
| [DATASET.md](mnist_project/DATASET.md) | Походження MNIST, формат, баланс класів і поділ |
| [LITERATURE.md](mnist_project/LITERATURE.md) | Огляд трьох наукових статей |
| [COMPARISON.md](mnist_project/COMPARISON.md) | Порівняння якості, швидкості та складності |
| [REQUIREMENTS.md](mnist_project/REQUIREMENTS.md) | Відповідність завданню |

### Навчання, оцінювання та результати

| Файл або папка | Призначення |
| --- | --- |
| [config.json](mnist_project/config.json) | Seeds, бюджет і умови експерименту |
| [data.py](mnist_project/data.py) | Завантаження MNIST, розбиття та передобробка |
| [models.py](mnist_project/models.py) | Архітектури MLP і CNN на шарах PyTorch |
| [experiment.py](mnist_project/experiment.py) | Навчання моделей і збереження checkpoint |
| [evaluate.py](mnist_project/evaluate.py) | Test-метрики, прогнози та стійкість |
| [benchmark.py](mnist_project/benchmark.py) | Повторний хронометраж single/batch прогнозування |
| [report.py](mnist_project/report.py) | Генерація звіту та графіків |
| [comparison_report.py](mnist_project/comparison_report.py) | Генерація порівняльної таблиці та висновків |
| [build_research.py](mnist_project/build_research.py) | Створення та виконання Notebook |
| [artifacts/](mnist_project/artifacts/) | Збережені результати експериментів |
| [checkpoints/](mnist_project/artifacts/checkpoints/) | Ваги дев’яти нейронних запусків |
| [logistic_regression.joblib](mnist_project/artifacts/logistic_regression.joblib) | Навчена логістична регресія |
| [model_comparison.csv](mnist_project/artifacts/model_comparison.csv) | Підсумкові числові результати трьох моделей |
| [training_summary.csv](mnist_project/artifacts/training_summary.csv) | Час навчання, параметри й validation-метрики |
| [benchmark_raw.csv](mnist_project/artifacts/benchmark_raw.csv) | Сирі повторні вимірювання часу |
| [benchmark_protocol.json](mnist_project/artifacts/benchmark_protocol.json) | Умови хронометражу, середовище й hashes ваг |
| [figures/](mnist_project/artifacts/figures/) | Приклади даних, криві навчання та графіки результатів |
| [tests/](mnist_project/tests/) | Перевірка даних, моделей, артефактів і застосунку |
| [requirements.txt](mnist_project/requirements.txt) | Python-залежності дослідження |

### Додаткова демонстрація

| Файл або папка | Призначення |
| --- | --- |
| [Публічний сайт MOMO](https://maryxseee.github.io/momo-number-studio/) | Розпізнавання намальованих цифр у браузері |
| [site/](site/) | Вихідний код браузерного застосунку |
| [docs/](docs/) | Статична копія для GitHub Pages |
| [WEB.md](WEB.md) | ONNX-модель, передобробка та технічне відтворення |
| [PERSONALIZATION.md](PERSONALIZATION.md) | Локальне донавчання та його обмеження |
| [tools/](tools/) | Експорт моделі, збірка й локальний перегляд сайту |
| [app.py](mnist_project/app.py) | Окрема локальна Python-демонстрація |
| [NUMBERS.md](mnist_project/NUMBERS.md) | Додатковий синтетичний benchmark чисел 1–1000 |

MNIST завантажується бібліотечним loader під час першого запуску; самі навчальні зображення й установлені бібліотеки до репозиторію не включено. Інструкції та команди — у [README дослідження](mnist_project/README.md).

Браузерний застосунок працює локально через ONNX Runtime Web. Персональні приклади й ваги зберігаються в браузері. Якість на MNIST не є виміряною точністю сайту чи персонального донавчання; мінуси, дробові роздільники та злиті цифри надійно не підтримуються.

Для локального перегляду сайту: `python tools/serve_site.py`, потім відкрити `http://127.0.0.1:8768`. Для оновлення статичної копії: `python tools/build_pages.py`.

### Попередній вступний експеримент

[digits_classification.ipynb](digits_classification.ipynb), [train_digits.py](train_digits.py) та [results/](results/) збережені як попередня робота на sklearn Digits. Основна робота для перевірки — MNIST у [mnist_project/](mnist_project/).
