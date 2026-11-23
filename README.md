# Analiza miar różnorodności próbek w baggingu drzew / lasach losowych

Projekt 1, Uczenie Zespołowe 2026/27 (J. Czarnecki, M. Dżaman).

Pytanie badawcze: czy miary różnorodności i reprezentatywności próbek uczących, obliczone
**przed** uczeniem drzew, wiążą się z jakością zespołu, także po uwzględnieniu siły drzew?

## Instalacja

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Dokładne wersje bibliotek: `requirements-lock.txt`, zapisane też w `results/<run>/config.json`.
Zbiory są pobierane z OpenML / UCI / scikit-learn przy pierwszym uruchomieniu i trafiają do
`data/cache/` (poza repozytorium).

## Uruchomienie

```bash
# eksperyment główny: 24 zbiory, CV 5x10, las losowy (domyślne parametry scikit-learn)
.venv/bin/python run_experiment.py --run main --n-repeats 10 --n-jobs 16

# czysty bagging (max_features=None) do porównania, tylko klasyfikacja
.venv/bin/python run_experiment.py --run main_bagging --task classification --ensemble bagging --n-repeats 10 --n-jobs 16

# eksperyment dodatkowy 4.6: 20 losowań klasycznego baggingu w 5 podziałach
.venv/bin/python run_experiment.py --run main_draws --draws 20 --n-jobs 16

# analiza statystyczna i wykresy
.venv/bin/python analyze.py --run main --bagging-run main_bagging --draws-run main_draws

# tabele i rysunki do raportu, kompilacja raportu (tectonic)
.venv/bin/python report/make_tables.py
cd report && tectonic -X compile raport.tex

# test zgodności wektorowych miar z implementacją referencyjną
.venv/bin/python tests/test_diversity_vectorized.py
```

Szybki test całego potoku: `--n-repeats 1 --n-estimators 10 --datasets sonar diabetes`.

Wyniki per fold: `results/<run>/folds/*.csv` (przerwane uruchomienie wznawia się od brakujących
foldów), wszystkie razem: `results/<run>/results.csv`, tabele analizy: `results/<run>/analysis/`,
wykresy: `figures/<run>/`. Ziarna są ustalone (`--seed 42`, podział k: 42 + k).

## Przygotowanie danych

Raz dla całego zbioru, przed podziałem (`src/datasets.py`): usunięcie kolumn-identyfikatorów i cech
znanych po fakcie, wierszy bez wartości docelowej, wierszy z brakami (zamiast imputacji), dokładnych
duplikatów, klas z < 5 obserwacjami, losowe ograniczenie do 20 000 obserwacji, usunięcie kolumn
stałych. Żaden z tych kroków nie korzysta ze statystyk liczonych na innych obserwacjach.

W każdym podziale, dopasowanie tylko na części treningowej (`src/preprocessing.py`,
`src/experiment.py`): kodowanie one-hot, kodowanie klas, standaryzacja dla miary R_X, przedziały
kwantylowe y dla S4. Raport z czyszczenia zwraca `load_dataset(...)["cleaning"]`.

## Struktura

| Plik | Zawartość |
|---|---|
| `src/datasets.py` | 12 + 12 zbiorów, cache, preprocessing na poziomie całego zbioru |
| `src/preprocessing.py` | preprocessing dopasowywany na części treningowej |
| `src/sampling.py` | warianty próbkowania S1–S4 |
| `src/diversity.py` | miary próbek ($D_J$, $D_C$, $D_Y$, $U$, $R_Y$, $R_X$), wersje parowe i wektorowe |
| `src/models.py` | drzewa, konfiguracje `random_forest` / `bagging`, agregacja |
| `src/experiment.py` | pętla eksperymentu i eksperyment dodatkowy 4.6 |
| `src/analysis.py` | testy H1–H3, porównania z kontrolą, test bayesowski, tabela mocy |
| `src/analysis_plots.py` | wykresy do raportu |
| `src/statistics.py`, `src/visualization.py` | wcześniejsza analiza i wykresy ogólne |
| `main.py`, `benchmark.py` | testy modułów i pomiar czasu |
| `report/` | raport (LaTeX), generator tabel `make_tables.py` |
