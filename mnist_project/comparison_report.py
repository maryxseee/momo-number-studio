"""Build the requirements-focused comparison from measured artifacts."""
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
NAMES = ['LogisticRegression', 'MLP', 'CNN']


def build_section():
    training = pd.read_csv(OUT / 'training_summary.csv')
    quality = pd.read_csv(OUT / 'clean_summary.csv').set_index('name')
    timing = pd.read_csv(OUT / 'benchmark_summary.csv').set_index(['name', 'mode'])
    records = []
    def uncertain(mean, std, digits):
        return f'{mean:.{digits}f}' + (f' ± {std:.{digits}f}' if pd.notna(std) else ' (1 запуск)')
    # Analytic multiply-accumulate estimates for Linear/Conv2d only; not runtime FLOPs.
    macs = {'LogisticRegression': 784*10,
            'MLP': 784*128 + 128*64 + 64*10,
            'CNN': 28*28*8*9 + 14*14*16*8*9 + 784*64 + 64*10}
    for name in NAMES:
        rows = training[training.name == name]
        paths = [OUT / ('logistic_regression.joblib' if name == 'LogisticRegression' else p)
                 for p in rows.checkpoint]
        records.append(dict(model=name, runs=len(rows), accuracy_pct=100*quality.loc[name, 'accuracy_mean'],
                            accuracy_std_pct=100*quality.loc[name, 'accuracy_std'],
                            macro_f1=quality.loc[name, 'macro_f1_mean'], parameters=int(rows.parameters.iloc[0]),
                            macs_per_image=macs[name], train_seconds_mean=rows.training_seconds.mean(),
                            train_seconds_std=rows.training_seconds.std(),
                            single_ms=timing.loc[(name, 'single'), 'ms_mean'],
                            single_ms_std=timing.loc[(name, 'single'), 'ms_std'],
                            batch_ms_per_image=timing.loc[(name, 'batch'), 'ms_mean'],
                            batch_ms_std=timing.loc[(name, 'batch'), 'ms_std'],
                            checkpoint_kib_mean=sum(p.stat().st_size for p in paths)/len(paths)/1024))
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / 'model_comparison.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in zip(axes, ['accuracy_pct', 'train_seconds_mean', 'batch_ms_per_image'],
                              ['Test accuracy (%)', 'Training time (s)', 'Batch inference (ms/image)']):
        error = {'accuracy_pct':'accuracy_std_pct', 'train_seconds_mean':'train_seconds_std',
                 'batch_ms_per_image':'batch_ms_std'}[col]
        ax.bar(frame.model, frame[col], yerr=frame[error].fillna(0), capsize=4,
               color=['#b99bd2', '#edb24c', '#d97d66'])
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=15)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylim(85, 100)
    axes[0].set_ylabel('Truncated scale: 85–100%')
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / 'efficiency_comparison.png', dpi=150)
    plt.close(fig)
    lines = ['| Модель | Accuracy, % | Macro-F1 | Навчання, с | 1 цифра, мс | Batch, мс/цифру | Параметри | Файл, KiB |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in records:
        lines.append(f"| {r['model']} | {uncertain(r['accuracy_pct'],r['accuracy_std_pct'],2)} | {r['macro_f1']:.4f} | "
                     f"{uncertain(r['train_seconds_mean'],r['train_seconds_std'],2)} | "
                     f"{uncertain(r['single_ms'],r['single_ms_std'],4)} | {uncertain(r['batch_ms_per_image'],r['batch_ms_std'],4)} | "
                     f"{r['parameters']} | {r['checkpoint_kib_mean']:.1f} |")
    fastest_single = frame.loc[frame.single_ms.idxmin(), 'model']
    fastest_batch = frame.loc[frame.batch_ms_per_image.idxmin(), 'model']
    fastest_train = frame.loc[frame.train_seconds_mean.idxmin(), 'model']
    gain = frame.iloc[2].accuracy_pct-frame.iloc[1].accuracy_pct
    ratio = frame.iloc[2].train_seconds_mean/frame.iloc[1].train_seconds_mean
    return '''## Порівняння трьох моделей: якість, швидкість і складність

Основні три різні моделі — LogisticRegression, MLP та CNN. CNN_aug — додаткова умова навчання CNN, а не четверта архітектура.

''' + '\n'.join(lines) + f'''

![Порівняння якості й витрат](artifacts/figures/efficiency_comparison.png)

У таблиці та error bars показано mean ± sample std між трьома seeds; для регресії один запуск і std між seeds не визначено. Для inference спочатку взято медіану 7 повторів кожного checkpoint. Це описовий розкид, а не довірчий інтервал. На графіку accuracy вісь починається з 85%, інші осі — з нуля.

### Протокол швидкодії

Навчання — фактичний час уже завершених запусків: середнє трьох seeds для нейромереж і один запуск регресії. Це історичні wall-clock вимірювання, без повторного хронометражу навчання; для нейромереж вони включають training loop, validation, запис checkpoint та історій і можуть залежати від фонового навантаження. Регресія використовує L-BFGS, нейромережі — 6 епох Adam, тому це ціна конкретних рецептів, а не однаковий оптимізаційний бюджет.

Прогнозування виміряне окремо скриптом `benchmark.py` на CPU: 2 потоки PyTorch та обмеження BLAS/OpenMP до 2 потоків, 3 прогрівальні batch-виклики та 7 повторів для кожного checkpoint. Медіана часу повторів обчислена всередині запуску, потім усереднена між seeds. Сім повторів часу не є сімома незалежними навчаннями. Сирі значення, квартилі та std між seeds збережені в CSV.

Режим «1 цифра» — 100 послідовних викликів batch=1 на перших 100 test-зображеннях; наведено середній час одного виклику всередині вимірювання. Batch — усі 10000 test-зображень блоками до 256; наведено амортизований час на зображення, а не затримку одного запиту. Вхід float32 [0,1] уже в RAM. Включено фіксовану нормалізацію нейромереж, predict_proba/softmax та argmax. Завантаження ваг, читання даних, сегментація й браузер не включені. Перед хронометражем прогнози кожного checkpoint звірено зі збереженими test-прогнозами; ваги не змінюються. Порядок моделей фіксований, результати залежать від процесора й навантаження; це не універсальний рейтинг швидкості.

Повне середовище та SHA-256 ваг — `artifacts/benchmark_protocol.json`; сирі часи — `benchmark_raw.csv`; медіани та квартилі — `benchmark_runs.csv`; підсумок — `model_comparison.csv`. Розмір файлу включає метадані серіалізації, для PyTorch також структуру checkpoint; це не пікова RAM і не чистий розмір ваг.

### Як визначено складність

Логістична регресія має один лінійний шар оцінок класів. MLP додає нелінійні повнозв'язні шари; CNN — дві згортки й pooling для просторових ознак. Параметрів відповідно 7850, 109386 та 52138. Отже, складніша архітектура не обов'язково має більше параметрів: CNN повторно використовує ті самі ядра у багатьох позиціях.

Аналітична оцінка multiply-accumulate для Linear/Conv2d на зображення: 7840, 109184 та 333056 MAC відповідно. Це лише оцінка цих шарів: bias, pooling, активації, softmax, переміщення пам'яті й накладні витрати бібліотек не враховані. Вона пояснює можливе зростання обчислень попри меншу кількість ваг CNN; не підміняє виміряний час.

### Практичний висновок

Найвища середня test accuracy серед основних трьох моделей — у CNN. Порівняно з MLP приріст становить {gain:.2f} відсоткового пункту, а історичне навчання тривало приблизно у {ratio:.1f} раза довше. Найменший виміряний середній час навчання — {fastest_train}; single-прогнозу — {fastest_single}; batch-прогнозування — {fastest_batch}. Розкид часу суттєвий; без контрольованого повторення на вільному процесорі це опис вимірювань, а не доведена стабільність рейтингу. Режими також відрізняються через накладні витрати та ефективність пакетних операцій.

Для простого baseline доречна регресія, для проміжного компромісу — MLP, для пріоритету якості в цьому протоколі — CNN. Вибір моделі для демонстрації здійснюється за validation macro-F1, а не за test. Додаткова аугментація оцінюється окремо за стійкістю. Ці вимірювання стосуються Python/CPU й не заявляються як швидкість або точність персональної браузерної моделі.
'''


if __name__ == '__main__':
    (ROOT / 'COMPARISON.md').write_text(build_section(), encoding='utf-8')
