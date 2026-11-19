"""Create and execute the research Notebook from the real experiment artifacts."""
import argparse
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent


def build(execute=False):
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text.strip()))

    md('''# Розпізнавання рукописних цифр: моделі та стійкість
**Дослідницький проєкт · MNIST · scikit-learn · PyTorch**

Автор: ____________________ · Група: ____________________

Мета: порівняти лінійний класифікатор, MLP та CNN і перевірити, як геометрична аугментація впливає на чисту якість і стійкість. Це навчання з нуля готових бібліотечних моделей, а не донавчання.

**Про виконання:** моделі фактично навчені скриптом `experiment.py`. Notebook показує їхній код, збережені історії та результати. За замовчуванням він повторно не витрачає час на навчання. Для повного відтворення встановіть `RETRAIN=True`; автоматичне перевикористання завершених запусків пояснене нижче.''')
    code('''from pathlib import Path
import sys, json, inspect, subprocess
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, Image, Code
ROOT = Path.cwd() if (Path.cwd()/'config.json').exists() else Path.cwd()/'mnist_project'
sys.path.insert(0, str(ROOT))
OUT = ROOT/'artifacts'
config = json.loads((ROOT/'config.json').read_text())
display(config)
''')
    md('''## Паспорт датасету та наукове підґрунтя
Основне завдання: на одному MNIST навчити три моделі різної архітектурної та обчислювальної складності — LogisticRegression, MLP, CNN — та порівняти якість і витрати. CNN_aug є додатковим експериментом, а не окремою архітектурою. Усна демонстрація сайту не заміняє це дослідження.''')
    code('''for document in ['DATASET.md', 'LITERATURE.md']:
    display(Markdown((ROOT/document).read_text(encoding='utf-8')))
''')
    md('''## 1. План до відкриття test
MNIST має 60000 офіційних train і 10000 test. Із train виділяємо 50000/10000 для навчання/валідації (stratified, seed=42). Для нейромереж seeds=17,42,73, шість епох, Adam, batch=256. Найкраща епоха визначається за validation macro-F1.

Гіпотеза 1: CNN перевершує логістичну регресію в цьому протоколі. Гіпотеза 2: аугментація покращує середню accuracy на +20°, −20° і зсуві 3 px. Чиста якість перевіряється окремо. Шум і товщина — додаткові stress tests, а не підстава змінювати модель.''')
    code('''from data import split_indices, augment_batch, CONDITIONS
display(Code(inspect.getsource(split_indices), language='python'))
if (OUT/'split.npz').exists():
    split = np.load(OUT/'split.npz')
    assert not set(split['train']) & set(split['validation'])
    print('Train:',len(split['train']),'Validation:',len(split['validation']),'Official test: 10000')
''')
    md('''## 2. Дані й підготовка
Вхід має форму [batch,1,28,28]. Для регресії зображення розгортається у 784 ознаки. Пікселі ділимо на 255; нейромережам додатково задаємо фіксований масштаб −1…1 без fitted statistics. Дані завантажує `torchvision.datasets.MNIST`.

Виконайте наступну клітинку для відтворення або використайте вже отримані артефакти. `experiment.py` відновлює роботу на рівні завершеного запуску. Щоб справді перенавчити всі моделі, спочатку перейменуйте папку `artifacts` як резервну копію. Не змінюйте протокол, спираючись на test-результати.''')
    code('''RETRAIN = False
if RETRAIN or not (OUT/'training_summary.csv').exists():
    subprocess.run([sys.executable,str(ROOT/'experiment.py')],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(ROOT/'evaluate.py')],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(ROOT/'benchmark.py')],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(ROOT/'report.py')],check=True,cwd=ROOT)
else:
    print('Використовуємо результати фактично завершеного навчання. Повторне навчання вимкнено.')
    if not (OUT/'evaluation.json').exists():
        subprocess.run([sys.executable,str(ROOT/'evaluate.py')],check=True,cwd=ROOT)
    if not (ROOT/'REPORT.md').exists() or not (OUT/'figures'/'reliability.png').exists():
        if not (OUT/'benchmark_summary.csv').exists():
            subprocess.run([sys.executable,str(ROOT/'benchmark.py')],check=True,cwd=ROOT)
        subprocess.run([sys.executable,str(ROOT/'report.py')],check=True,cwd=ROOT)
display(json.loads((OUT/'environment.json').read_text()))
''')
    code('''display(Image(filename=str(OUT/'figures'/'samples.png')))
display(pd.read_csv(OUT/'class_balance.csv'))
display(Image(filename=str(OUT/'figures'/'class_balance.png')))
''')
    md('''## 3. Архітектури
MLP порівнюємо зі згортковою мережею. CNN і CNN_aug мають однакову архітектуру; змінюється лише навчальна передобробка. Dropout активний під час навчання та вимкнений під час оцінювання.''')
    code('''from models import make_model
import torch
display(Code(inspect.getsource(make_model),language='python'))
for kind in ['mlp','cnn']:
    model=make_model(kind)
    print(kind,':',sum(p.numel() for p in model.parameters()),'параметрів; logits',tuple(model(torch.zeros(2,1,28,28)).shape))
''')
    md('''## 4. Реалізація навчання
Нижче наведено фактичний training loop із модуля `experiment.py`: `zero_grad → forward → CrossEntropyLoss → backward → optimizer.step`. Backpropagation виконує PyTorch autograd. CrossEntropyLoss отримує logits, без попереднього softmax. Після кожної епохи модель перевіряється на validation; test у функцію навчання не передається.''')
    code('''import experiment
display(Code(inspect.getsource(experiment.evaluate_validation),language='python'))
display(Code(inspect.getsource(experiment.run),language='python'))
''')
    md('''## 5. Історії та вибір checkpoint
Графіки показують mean ± sample std трьох запусків. Train метрики отримані в train-режимі під час оновлень ваг; validation — в eval після епохи. Для CNN_aug train також спотворений. Тому пряме порівняння train і validation має це обмеження. Всі шість епох виконуються; збереження найкращої епохи не є early stopping.''')
    code('''training=pd.read_csv(OUT/'training_summary.csv')
display(training[['name','seed','validation_macro_f1','selected_epoch','parameters','training_seconds']].round(4))
display(pd.read_csv(OUT/'history_CNN_aug_42.csv').round(4))
display(Image(filename=str(OUT/'figures'/'loss_curves.png')))
display(Image(filename=str(OUT/'figures'/'accuracy_curves.png')))
''')
    md('''## 6. Контрольований експеримент з аугментацією
Поворот ±15°, зсув ±10%, масштаб 0.9…1.1 — незалежно для кожного зображення. Перетворення реалізовані бібліотечними операціями. У парі з однаковим seed CNN і CNN_aug починають із тих самих ваг; перевіримо це за hashes. Shuffle і аугментація використовують окремі generators.''')
    code('''display(Code(inspect.getsource(augment_batch),language='python'))
for seed in config['seeds']:
    plain=json.loads((OUT/f'CNN_{seed}.json').read_text())
    aug=json.loads((OUT/f'CNN_aug_{seed}.json').read_text())
    assert plain['initial_weights_sha256']==aug['initial_weights_sha256']
    print(seed,': початкові ваги однакові')
''')
    md('''## 7. Незалежне оцінювання на чистому test
Accuracy — частка правильних прогнозів. Macro-F1 — середнє F1 класів, які мають рівну вагу. Результати всіх seeds показано, а не відібрано за найкращим test. Для регресії є один детермінований запуск; std не визначено. Test-прогнози можна перевірити у `predictions.npz`.''')
    code('''evaluation=json.loads((OUT/'evaluation.json').read_text())
clean=pd.read_csv(OUT/'clean_summary.csv')
display(clean.round(5))
display(Image(filename=str(OUT/'figures'/'clean_comparison.png')))
predictions=np.load(OUT/'predictions.npz')
for run in evaluation['runs']:
    key=f"{run['name']}_{run['seed']}__clean"
    actual=float((predictions[key]==predictions['truth']).mean())
    assert abs(actual-run['conditions']['clean']['accuracy'])<1e-12
print('Збережені прогнози узгоджені з accuracy всіх 10 запусків.')
''')
    md('''## 8. Наперед задані stress tests
±20° і зсув праворуч 3 px перевіряють перенесення геометричної стійкості. Gaussian noise σ=0.2 та dilation/erosion 3×3 — інші типи зміни розподілу. Кожне спотворення застосовується окремо до чистого test; моделі отримують ідентичні входи й не перенавчаються.''')
    code('''display(Image(filename=str(OUT/'figures'/'corruption_examples.png')))
display(Image(filename=str(OUT/'figures'/'robustness.png')))
aggregate=pd.read_csv(OUT/'aggregated.csv')
display(aggregate.pivot(index='name',columns='condition',values='accuracy_mean').round(4))
''')
    md('''## 9. Парні різниці та перевірка основної гіпотези
Спочатку обчислюємо CNN_aug − CNN для кожного seed, потім усереднюємо. Основний ефект — середнє трьох геометричних умов; чиста якість — окремий компроміс. Відсоткові пункти не слід плутати з відносними відсотками. Std між трьома seeds не є довірчим інтервалом.''')
    code('''paired=pd.read_csv(OUT/'paired_deltas.csv')
display(paired.round(5))
primary=paired[paired.condition.isin(['rotate_plus20','rotate_minus20','shift_right3'])].groupby('seed').delta_accuracy.mean()
clean_effect=paired[paired.condition=='clean'].delta_accuracy
print(f'Геометрія: mean delta={primary.mean()*100:+.2f} в.п.; std={primary.std(ddof=1)*100:.2f} в.п.')
print(f'Чисті дані: mean delta={clean_effect.mean()*100:+.2f} в.п.')
display(Image(filename=str(OUT/'figures'/'augmentation_deltas.png')))
''')
    md('''## 10. Аналіз помилок
Для кожного сімейства представник обирається за validation. Нижче — матриці помилок і перші неправильні прогнози моделі демонстрації в офіційному порядку test. У таблиці для обраного checkpoint наведено precision, recall і F1 кожної цифри.''')
    code('''selected=json.loads((OUT/'demo_selection.json').read_text())
display(selected)
display(pd.read_csv(OUT/f"classes_{selected['name']}_{selected['seed']}.csv",index_col=0).round(4))
display(Image(filename=str(OUT/'figures'/'confusion_matrices.png')))
display(Image(filename=str(OUT/'figures'/'errors.png')))
''')
    md('''## 11. Оцінки softmax і невизначеність
Softmax scores не є гарантованою впевненістю. Reliability diagram та 10-bin ECE — описова діагностика, без калібрування на test. Wilson-інтервал accuracy припускає незалежність test-прикладів і не враховує варіативність навчання чи зміну домену.''')
    code('''chosen=next(r for r in evaluation['runs'] if r['name']==selected['name'] and r['seed']==selected['seed'])
print('Accuracy 95% Wilson interval:',chosen['conditions']['clean']['accuracy_wilson_95'])
display(json.loads((OUT/'diagnostics.json').read_text()))
display(Image(filename=str(OUT/'figures'/'reliability.png')))
''')
    md('''## 12. Демонстрація та висновки
Запуск у терміналі: `python app.py`, потім відкрити `http://127.0.0.1:8765`.
Полотно обрізається, вписується в 20×20, центрується в 28×28; користувач бачить вхід моделі й top-3. Порожнє полотно відхиляється. Це демонстрація, а не зовнішній розмічений тест.

Нижче завантажується повний звіт, автоматично сформований із фактичних результатів. Числа не підставлені вручну.''')
    code('''display(Markdown((ROOT/'REPORT.md').read_text(encoding='utf-8')))
''')
    md('''## 13. Розпізнавання чисел 1–1000
Для одного числа в рядок із проміжками повторно використовуємо digit CNN. Виділяємо цифри за вертикальною проєкцією, об'єднуємо дрібні внутрішні розриви, упорядковуємо зліва направо й перевіряємо ціле число. Неприпустимий прогноз не підміняється дозволеним. Порожній ввід і понад чотири сегменти відхиляються.

На головній сторінці застосунку тепер широке полотно; попередній режим однієї цифри доступний за `/digit`.''')
    code('''from inference import segment_number, predict_number, validate_number
display(Code(inspect.getsource(segment_number),language='python'))
display(Code(inspect.getsource(predict_number),language='python'))
for text in ['407','1000','1001','01','0']:
    print(text,'→',validate_number(text))
''')
    md('''## 14. Синтетичний benchmark цілих чисел
Для всіх 1…1000 складено test-glyphs MNIST у двох фіксованих умовах: 18 px між рамками та нульовий проміжок. Контроль із відомими межами отримує те саме очищення та preprocessing. Це не справжні багатозначні рукописні зразки й не новий незалежний test: glyphs повторюються, джерело вже використане в основному експерименті.''')
    code('''if not (OUT/'numbers'/'summary.json').exists():
    subprocess.run([sys.executable,str(ROOT/'evaluate_numbers.py')],check=True,cwd=ROOT)
number_summary=json.loads((OUT/'numbers'/'summary.json').read_text())
display(pd.DataFrame(number_summary['layouts']).T)
display(Image(filename=str(OUT/'numbers'/'examples.png')))
from number_report import build_section
display(Markdown(build_section()))
''')
    md('''## 15. Швидкодія та підсумок трьох моделей
Повторний benchmark використовує збережені ваги й не навчає моделі. Якщо артефакти вже є, Notebook показує фактичні вимірювання; для нового хронометражу виконайте `python benchmark.py`, потім `python report.py`. Наведені часи стосуються CPU/Python, а не сайту.''')
    code('''if not (OUT/'benchmark_summary.csv').exists():
    subprocess.run([sys.executable,str(ROOT/'benchmark.py')],check=True,cwd=ROOT)
from comparison_report import build_section as build_comparison
display(pd.read_csv(OUT/'benchmark_runs.csv').round(6))
display(Markdown(build_comparison()))
''')
    md('''## Відповідність завданню
Обрана задача — класифікація; дані — MNIST; навчання — sklearn і PyTorch; оцінювання — validation/test, accuracy, macro-F1, per-class report, матриці, повторні запуски й stress tests. Немає реалізації ML-алгоритмів або градієнтів з нуля. Додатково надані модулі, ваги, протокол, графіки, звіт і локальний застосунок.

Джерела: [MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html), [PyTorch training](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html), [scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html).''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
    nbf.validate(nb)
    if execute:
        NotebookClient(nb,timeout=1800,resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(nb,ROOT/'research.ipynb')
    print(f'Notebook: {len(cells)} cells, executed={execute}')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    build(parser.parse_args().execute)
