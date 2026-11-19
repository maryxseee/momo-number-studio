"""Generate publication-style experiment plots and a Ukrainian factual report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torchvision.datasets import MNIST
from data import CONDITIONS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
FIG = OUT / 'figures'
NAMES = ['LogisticRegression', 'MLP', 'CNN', 'CNN_aug']
LABELS = ['Чисті', '+20°', '−20°', 'Зсув 3 px', 'Шум σ=0.2', 'Товсті', 'Тонкі']
COLORS = ['#64748b', '#0891b2', '#2563eb', '#e07a24']


def table(frame):
    f = frame.copy()
    for col in f.columns:
        f[col] = f[col].map(lambda x: f'{x:.4f}' if isinstance(x, (float, np.floating)) else str(x))
    return '\n'.join(['| ' + ' | '.join(f.columns) + ' |', '| ' + ' | '.join(['---']*len(f.columns)) + ' |'] + ['| '+' | '.join(row)+' |' for row in f.values.tolist()])


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close()


def run():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'figure.dpi':110, 'axes.spines.top':False, 'axes.spines.right':False})
    evaluation = json.loads((OUT/'evaluation.json').read_text())
    training = pd.read_csv(OUT/'training_summary.csv')
    aggregate = pd.read_csv(OUT/'aggregated.csv')
    paired = pd.read_csv(OUT/'paired_deltas.csv')
    config = json.loads((OUT/'protocol.json').read_text())
    selected = json.loads((OUT/'demo_selection.json').read_text())
    predictions = np.load(OUT/'predictions.npz')
    clean = aggregate[aggregate.condition=='clean'].set_index('name').reindex(NAMES).reset_index()
    clean.to_csv(OUT/'clean_summary.csv', index=False)

    # Examples are drawn from the training source, never used for test-driven tuning.
    train = MNIST(ROOT/'data', train=True, download=False)
    split = np.load(OUT/'split.npz')
    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for digit in range(10):
        candidates = [int(i) for i in split['train'] if int(train.targets[i]) == digit][:2]
        for row, i in enumerate(candidates):
            axes[row,digit].imshow(train.data[i], cmap='gray', vmin=0, vmax=255)
            axes[row,digit].axis('off')
            if row == 0:
                axes[row,digit].set_title(str(digit))
    fig.suptitle('Приклади MNIST із навчальної частини')
    save('samples')
    counts = pd.DataFrame({'train':np.bincount(train.targets[split['train']], minlength=10), 'validation':np.bincount(train.targets[split['validation']], minlength=10), 'test':np.bincount(predictions['truth'], minlength=10)})
    counts.to_csv(OUT/'class_balance.csv', index_label='digit')
    counts.plot.bar(figsize=(10, 4), rot=0, color=['#2563eb','#0891b2','#e07a24'])
    plt.xlabel('Цифра'); plt.ylabel('Кількість'); plt.title('Розподіл класів у трьох частинах')
    save('class_balance')

    for metric, filename, ylabel in [('loss','loss_curves','Cross-entropy'), ('accuracy','accuracy_curves','Accuracy')]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, name in zip(axes, NAMES[1:]):
            frames = [pd.read_csv(OUT/f'history_{name}_{seed}.csv') for seed in config['seeds']]
            epochs = frames[0].epoch
            for prefix, color, label in [('train','#64748b','Навчання'),('validation','#2563eb','Валідація')]:
                values = np.stack([f[f'{prefix}_{metric}'].values for f in frames])
                mean, std = values.mean(0), values.std(0, ddof=1)
                ax.plot(epochs, mean, 'o-', label=label, color=color)
                ax.fill_between(epochs, mean-std, mean+std, color=color, alpha=.12)
            ax.set(title=name, xlabel='Епоха', ylabel=ylabel)
            ax.legend(fontsize=8)
        fig.suptitle('Історії навчання: середнє ± sample std трьох seeds')
        save(filename)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in zip(axes, ['accuracy','macro_f1'], ['Accuracy','Macro-F1']):
        ax.bar(clean.name, clean[f'{metric}_mean'], yerr=clean[f'{metric}_std'].fillna(0), color=COLORS, capsize=5)
        ax.set(ylim=(0.85,1), ylabel=title, title=f'Чистий test: {title}')
        ax.tick_params(axis='x', labelrotation=15)
    save('clean_comparison')
    heat = aggregate.pivot(index='name', columns='condition', values='accuracy_mean').reindex(index=NAMES, columns=CONDITIONS)
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(heat.values, vmin=0, vmax=1, cmap='YlGnBu')
    ax.set_xticks(range(7), LABELS, rotation=15)
    ax.set_yticks(range(4), NAMES)
    for i in range(4):
        for j in range(7):
            ax.text(j,i,f'{100*heat.iloc[i,j]:.1f}%',ha='center',va='center',color='white' if heat.iloc[i,j]>.6 else 'black')
    ax.set_title('Accuracy за моделями й наперед заданими спотвореннями')
    fig.colorbar(im, ax=ax, label='Accuracy')
    save('robustness')

    deltas = paired.groupby('condition').delta_accuracy.agg(['mean','std']).reindex(CONDITIONS)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(LABELS, deltas['mean']*100, yerr=deltas['std']*100, capsize=4, color='#0891b2')
    ax.axhline(0, color='#475569', linewidth=1)
    ax.set(ylabel='Зміна accuracy, відсоткові пункти', title='CNN_aug − CNN: парні різниці, mean ± sample std (n=3)')
    ax.tick_params(axis='x', labelrotation=15)
    save('augmentation_deltas')
    examples = np.load(OUT/'corruption_examples.npz')
    fig, axes = plt.subplots(2, 7, figsize=(12, 4))
    for j, condition in enumerate(CONDITIONS):
        for i in range(2):
            axes[i,j].imshow(examples[condition][i,0], cmap='gray', vmin=0,vmax=1)
            axes[i,j].axis('off')
        axes[0,j].set_title(LABELS[j])
    fig.suptitle('Однакові приклади за різних умов тестування')
    save('corruption_examples')

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    representative = {}
    for ax, name in zip(axes.flat, NAMES):
        candidates = [r for r in evaluation['runs'] if r['name']==name]
        run = max(candidates, key=lambda r:r['validation_macro_f1'])
        representative[name] = run
        ConfusionMatrixDisplay(np.array(run['confusion_matrix'])).plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
        ax.set(title=f'{name}, seed={run["seed"]}', xlabel='Прогноз', ylabel='Істина')
    fig.suptitle('Представники кожної моделі обрані за validation')
    save('confusion_matrices')

    key = f'{selected["name"]}_{selected["seed"]}__clean'
    prediction, truth = predictions[key], predictions['truth']
    confidence = predictions[key+'__confidence']
    wrong = np.flatnonzero(prediction != truth)
    test = MNIST(ROOT/'data', train=False, download=False)
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    for ax in axes.flat:
        ax.axis('off')
    for ax, index in zip(axes.flat, wrong[:12]):
        ax.imshow(test.data[int(index)], cmap='gray')
        ax.set_title(f'{truth[index]} → {prediction[index]} · score {confidence[index]:.2f}')
    fig.suptitle(f'Перші помилки моделі демонстрації: {selected["name"]}, seed={selected["seed"]}')
    save('errors')
    # Equal-width reliability bins: descriptive diagnostic, no post-test calibration.
    calibration = []
    for i in range(10):
        mask = (confidence >= i/10) & ((confidence < (i+1)/10) if i<9 else (confidence<=1))
        if mask.any():
            calibration.append({'bin':i, 'n':int(mask.sum()), 'confidence':float(confidence[mask].mean()), 'accuracy':float((prediction[mask]==truth[mask]).mean())})
    calibration = pd.DataFrame(calibration)
    calibration.to_csv(OUT/'reliability.csv', index=False)
    ece = float((calibration.n * abs(calibration.confidence-calibration.accuracy)).sum()/len(truth))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0,1],[0,1],'--',color='#94a3b8',label='Ідеальна відповідність')
    ax.scatter(calibration.confidence, calibration.accuracy, s=30+calibration.n/30, color='#2563eb')
    ax.set(xlabel='Середня оцінка softmax у bin',ylabel='Частка правильних відповідей', title=f'Reliability diagram · ECE={ece:.4f}',xlim=(0,1),ylim=(0,1.03))
    ax.legend()
    save('reliability')
    (OUT/'diagnostics.json').write_text(json.dumps({'demo_errors':len(wrong), 'demo_ece_10bins':ece},indent=2))

    clean_table = table(clean[['name','n','accuracy_mean','accuracy_std','macro_f1_mean','macro_f1_std']].astype(object).fillna('—'))
    delta_table = table(deltas.reset_index().rename(columns={'mean':'delta_accuracy_mean','std':'delta_accuracy_std'}))
    primary = paired[paired.condition.isin(['rotate_plus20','rotate_minus20','shift_right3'])].groupby('seed').delta_accuracy.mean()
    clean_delta = paired[paired.condition=='clean'].delta_accuracy
    cnn = clean.set_index('name').loc['CNN']
    aug = clean.set_index('name').loc['CNN_aug']
    lr = clean.set_index('name').loc['LogisticRegression']
    environment = json.loads((OUT/'environment.json').read_text())
    selected_result = next(r for r in evaluation['runs'] if r['name']==selected['name'] and r['seed']==selected['seed'])
    interval = selected_result['conditions']['clean']['accuracy_wilson_95']
    outcome = 'покращила' if primary.mean()>0 else 'не покращила'
    text = f'''# Розпізнавання рукописних цифр: порівняння моделей і дослідження аугментації

Автор: Колесник Марія · КН-3, група 1

## Анотація

Досліджено багатокласову класифікацію зображень цифр на MNIST. Порівняно логістичну регресію, багатошаровий перцептрон (MLP), згорткову нейромережу (CNN) і ту саму CNN з геометричною аугментацією. Кожний нейронний варіант навчено з трьома seeds. Вибір checkpoint здійснено за validation macro-F1, а test використано для фінального оцінювання. Усі наведені числа отримано виконанням коду, а не задано вручну. CNN без аугментації отримала середню accuracy {cnn.accuracy_mean:.2%}, CNN з аугментацією — {aug.accuracy_mean:.2%}. На трьох заздалегідь визначених геометричних спотвореннях середня парна зміна accuracy становила {primary.mean()*100:+.2f} відсоткового пункту. Це результат обмеженого обчислювального експерименту, а не гарантія якості на довільних рукописних зображеннях.

## 1. Постановка задачі та гіпотези

Вхід — одноканальне зображення 28×28; вихід — один із класів 0–9. Це supervised multiclass classification. Перша гіпотеза: локальна структура зображення, яку враховує CNN, дає кращу якість за лінійний класифікатор при нашому бюджеті навчання. Друга, основна гіпотеза: геометрична аугментація покращує стійкість CNN до поворотів і зсувів. Окремо вимірюється ціна такої стійкості на чистих даних. Основна узагальнювальна stress-метрика — рівнозважене середнє accuracy для +20°, −20° і зсуву праворуч 3 px; вона не є оцінкою частоти цих умов у реальному житті. Шум і товщина штриха — додаткові, а не навчальні перетворення.

## 2. Дані й запобігання витоку

Офіційна навчальна частина MNIST має 60000 прикладів. Стратифікованим поділом із seed=42 виділено 50000 train і 10000 validation. Офіційні 10000 test не включено ні в навчання, ні у вибір епохи. Усі моделі використовують однакові індекси. Пікселі переводяться з 0…255 у 0…1; для нейромереж додаткове фіксоване перетворення переводить їх у −1…1. Середні або дисперсії на validation/test не оцінюються. Аугментація застосовується виключно до навчальних mini-batches; validation лишається чистою. Індекси розбиття та SHA-256 завантажених архівів збережено. Відтворюваність випадкового поділу не означає незалежності авторів почерку.

![Дані](artifacts/figures/samples.png)

## 3. Моделі та навчання

Логістична регресія: C=1, L-BFGS, max_iter=300, tol=0.001, 784 вхідні ознаки. Це сильніший орієнтир, ніж випадковий прогноз; підбір параметрів цієї моделі не проводився. Модель sklearn отримує значення 0…1 без окремого StandardScaler.

MLP: Flatten → Linear(784,128) → ReLU → Dropout(0.2) → Linear(128,64) → ReLU → Linear(64,10).

CNN: Conv(1,8,3×3,padding=1) → ReLU → MaxPool(2) → Conv(8,16,3×3,padding=1) → ReLU → MaxPool(2) → Flatten → Linear(784,64) → ReLU → Dropout(0.2) → Linear(64,10). CNN_aug має ідентичну архітектуру.

Для всіх нейромереж: Adam, learning rate 0.001, weight_decay 0.0001, batch=256, шість епох, seeds 17/42/73. Функція втрат — CrossEntropyLoss над logits; softmax перед нею не застосовується. Градієнти обчислює autograd; крок оптимізації виконує готовий Adam. Для кожного запуску збережено checkpoint епохи з найбільшим validation macro-F1 (при рівності — раніша). Навчання триває всі шість епох: це вибір checkpoint, а не early stopping.

Для пар CNN/CNN_aug із тим самим seed початкові ваги ідентичні; їхні hashes збережено. DataLoader має незалежний generator для однакового порядку прикладів. Аугментація використовує окремий generator і не змінює потік випадковості dropout. Результати можуть залежати від бібліотек і процесора, тому версії зафіксовано. Використано CPU і два потоки PyTorch; однакова кількість епох не означає однакову кількість операцій чи час.

![Втрати](artifacts/figures/loss_curves.png)

Train loss вимірюється під час оновлень ваг із активним dropout (і аугментацією для CNN_aug); validation loss — після епохи в eval-режимі на чистих зображеннях. Тому ці криві не є ідентичними умовами вимірювання, і їхній розрив не слід автоматично називати перенавчанням.

## 4. Аугментація та контрольовані спотворення

Під час навчання CNN_aug кожне зображення незалежно отримує випадковий поворот у ±15° та масштаб 0.9…1.1. Параметр зсуву оберненої нормалізованої сітки змінюється в ±0.2: без поєднання з масштабом і поворотом це відповідає приблизно ±10% ширини. За спільної афінної трансформації фактичний прямий зсув може бути більшим, тому test-зсув 3 px не оголошуємо поза навчальним діапазоном. Перетворення виконується бібліотечними affine_grid/grid_sample з bilinear interpolation і чорним фоном. Test-повороти ±20° виходять за навчальні ±15°. Це перевірка обмеженого перенесення стійкості, а не повторення тих самих навчальних картинок.

Додаткові умови: адитивний Gaussian noise σ=0.2 у шкалі 0…1 з обрізанням до [0,1]; dilation та erosion квадратним ядром 3×3. Такі морфологічні перетворення можуть бути сильними й змінювати читабельність, тому їхній результат не слід трактувати як стійкість до будь-якого природного почерку. Всі моделі отримують однакові спотворені зображення. Кожна умова є окремою і не накладається поверх попередньої. Моделі після test не переналаштовувалися.

![Спотворення](artifacts/figures/corruption_examples.png)

## 5. Метрики та чистий test

Accuracy — частка правильних відповідей. Precision і recall обчислюються для кожної цифри; F1 — їх гармонічне середнє. Macro-F1 надає класам однакову вагу. Таблиця містить mean і sample std по трьох нейронних seeds; для єдиного запуску регресії std не визначено. Std між seeds описує варіативність навчання на фіксованому поділі, а не довірчий інтервал якості на всіх можливих даних.

{clean_table}

![Чистий test](artifacts/figures/clean_comparison.png)

Середня різниця CNN − LogisticRegression на чистому test: {(cnn.accuracy_mean-lr.accuracy_mean)*100:+.2f} відсоткового пункту. Це порівняння конкретних архітектур і фіксованих налаштувань, а не доказ загальної переваги всіх нейронних мереж над усіма класичними алгоритмами. Повні per-class precision/recall/F1 збережені у CSV для кожного запуску.

## 6. Основний експеримент: ефект аугментації

{delta_table}

У таблиці різниці в частках (0.01 = один відсотковий пункт). Парні різниці рахуються спочатку всередині seed, потім усереднюються. За основною геометричною stress-метрикою аугментація **{outcome}** accuracy: mean Δ={primary.mean()*100:+.2f} в.п., sample std={primary.std(ddof=1)*100:.2f} в.п. На чистому test mean Δ={clean_delta.mean()*100:+.2f} в.п. Три seeds не дають підстав стверджувати статистичну значущість або гарантувати знак ефекту на іншому поділі.

![Ефект аугментації](artifacts/figures/augmentation_deltas.png)

![Стійкість](artifacts/figures/robustness.png)

## 7. Аналіз помилок та оцінок softmax

Представника кожного сімейства для матриць помилок обрано за validation, без пошуку найкращого test seed. Для демонстрації за тим самим правилом обрано {selected['name']}, seed={selected['seed']}, епоха={selected['selected_epoch']}. На чистому test він має {len(wrong)} помилок. Для його accuracy Wilson 95% interval: [{interval[0]:.2%}; {interval[1]:.2%}] за припущення незалежних прикладів; цей інтервал не враховує навчальну випадковість або зміну домену.

![Матриці](artifacts/figures/confusion_matrices.png)

![Помилки](artifacts/figures/errors.png)

Показано перші помилки в офіційному порядку test, а не підібрані вручну «гарні» приклади. Візуальна неоднозначність є можливою інтерпретацією, але не встановленою причиною. Діаграма надійності зі сталими десятьма bins дала ECE={ece:.4f}; це лише описова, залежна від bins діагностика обраної моделі. Жодне калібрування на test не проводилося. Навіть високий softmax score не гарантує правильності для малюнка з іншого джерела.

![Reliability](artifacts/figures/reliability.png)

## 8. Практична демонстрація

Локальний застосунок app.py приймає малюнок із canvas, обрізає порожній фон, вписує штрих у поле 20×20, доповнює до 28×28 і центрує за центром мас. Після тієї самої фіксованої нормалізації модель повертає top-3 softmax scores і зображення, яке вона отримала. Порожні та некоректні входи відхиляються. Модель завантажується зі збереженого checkpoint, сервер слухає лише 127.0.0.1. Малюнки не зберігаються. Власний UI — демонстрація, а не розмічений зовнішній benchmark; точність на намальованих користувачем цифрах тут не оцінювалася.

## 9. Обмеження та подальша робота

1. Шість епох — фіксований CPU-бюджет, а не доведена збіжність кожної моделі. CNN_aug може потребувати довшого навчання; висновки обмежені цим бюджетом.
2. Один train/validation split і три seeds оцінюють лише частину невизначеності. Повторні розбиття та незалежні автори почерку підсилили б висновки.
3. MNIST містить підготовлені зображення; фотографії мають інший фон, світло, перспективу. Для них потрібні дані й окреме оцінювання.
4. Геометрична аугментація не навчає модель автоматично стійкості до шуму чи морфології. Результати для цих умов потрібно читати окремо.
5. Логістична регресія не мала пошуку гіперпараметрів, а нейромережі мають одну архітектуру й оптимізатор. Порівняння не вичерпує класів моделей.
6. ECE залежить від розбиття на bins; Wilson interval припускає незалежність test-прикладів; жодна метрика не заміняє перевірки нового домену.

Подальші експерименти: більший наперед визначений бюджет, кілька незалежних split, контроль кожного типу аугментації окремо, оцінювання на розміченому наборі малюнків нових авторів. Вони не видаються за виконані результати цієї роботи.

## 10. Висновки

Реалізовано повний цикл: отримання даних, контрольований поділ, навчання бібліотечних моделей, вибір checkpoint за validation, незалежне test-оцінювання, парне дослідження аугментації, аналіз помилок і інтерактивне застосування. Основний результат — не лише accuracy, а виміряний компроміс чистої якості та стійкості: геометричний ефект {primary.mean()*100:+.2f} в.п. при зміні чистої accuracy на {clean_delta.mean()*100:+.2f} в.п. Всі цифри можна перевірити за збереженими прогнозами, історіями та вагами.

## Джерела та відтворення

- [Torchvision MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html).
- [PyTorch: навчання моделей](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).
- [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
- [Grid sample](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).
- [scikit-learn: оцінювання](https://scikit-learn.org/stable/modules/model_evaluation.html).

Середовище фактичного запуску: {environment['versions']}. Команди відтворення наведені в README. Артефакти містять конфігурацію, SHA-256 даних і checkpoint, індекси, історії, всі test-прогнози, таблиці та рисунки.
'''
    from number_report import build_section
    number_section=build_section()
    if number_section:
        text=text.replace('## Джерела та відтворення',number_section+'\n\n## Джерела та відтворення')
    from comparison_report import build_section as build_comparison
    comparison = build_comparison()
    (ROOT/'COMPARISON.md').write_text(comparison, encoding='utf-8')
    background = '\n\n'.join((ROOT/name).read_text(encoding='utf-8') for name in ['DATASET.md', 'LITERATURE.md'])
    text = text.replace('## 1. Постановка задачі та гіпотези', background+'\n\n## 1. Постановка задачі та гіпотези')
    text = text.replace('## 6. Основний експеримент:', comparison+'\n\n## 6. Основний експеримент:')
    (ROOT/'REPORT.md').write_text(text, encoding='utf-8')
    print('Report and figures generated.', flush=True)


if __name__ == '__main__':
    run()
