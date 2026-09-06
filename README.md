# Premier League Transfer Value Predictor

A learning project: predict a Premier League player's market value (EUR)
from simple season stats (goals, assists, minutes, age, position...), then
compare the prediction against the player's actual market value. Built to
learn regression fundamentals - log-transforming a skewed target, why a
random train/test split leaks information here, reading linear model
coefficients under collinearity, and where a linear model is simply the
wrong tool.

Data: [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)
(CC0). Scope: Premier League only (`competition_id == "GB1"`), last 10
completed seasons (a full decade: 2016-17 through 2025-26).

## How to run

```bash
pip install -r requirements.txt

python -m src.fetch            # download + cache raw CSVs into data/raw/
python -m src.build_dataset    # -> data/processed/player_seasons.csv
python -m src.train            # trains all 4 models, writes reports/metrics.json
python -m src.visualize        # writes 4 PNG charts to reports/

python -m src.predict "Bukayo Saka" --season 2024
python -m src.predict "Erling Haaland" --season 2025 --model xgboost

pytest -q                      # unit tests for feature engineering + age calc
```

Re-running `fetch`/`build_dataset`/`train` is safe and idempotent - raw CSVs
are cached on disk and only re-downloaded with `force=True`.

## Pipeline

| Module | Responsibility |
|---|---|
| `config.py` | Every path, season, ID, and threshold - the single source of truth |
| `src/fetch.py` | Download + cache the 5 raw CSVs |
| `src/build_dataset.py` | Aggregate appearances to one row per player per season, join player attributes and the market value *current at that season's end* |
| `src/features.py` | Filter low-minute rows, engineer `*_per_90`, `age_squared`, log1p the target |
| `src/train.py` | Fit LinearRegression / RidgeCV / RandomForest / XGBoost inside a shared `Pipeline`, evaluate on the held-out season, persist models |
| `src/predict.py` | CLI: look up one player-season's predicted vs actual value |
| `src/visualize.py` | The 4 required matplotlib charts |

## Results (test season: 2025, trained on 2016-2024)

| Model | MAE (EUR) | RMSE (EUR) | R² (log-value scale) |
|---|---:|---:|---:|
| Position-median baseline | 17,097,890 | 26,274,706 | -0.038 |
| Linear Regression | 12,820,811 | 19,543,224 | 0.482 |
| **Ridge** (α≈0.032, 5-fold CV) | **12,820,757** | **19,543,142** | **0.482** |
| Random Forest | 12,922,338 | 20,070,577 | 0.443 |
| XGBoost | 12,550,113 | 19,708,223 | 0.497 |

XGBoost was added as a fourth model - unlike Ridge's coefficients, its
predictions aren't directly interpretable from the model itself, which is
exactly why SHAP-based explanations exist; see the comparison paragraph
below for whether the accuracy gain is actually worth that tradeoff.

R² is reported on the log1p scale because that's the scale the models were
actually fit on and optimized for; an R² computed on raw euros would be
dominated by the handful of 100M+ outliers and would mostly just measure
"did you get Haaland roughly right," not overall fit quality.

**The stats-only model clears the naive baseline by a wide margin** (R² 0.48
vs -0.04 - the baseline is now *worse than predicting the training mean*)
- so season performance genuinely carries signal about market value, not
just "attackers are worth more than defenders." Ridge and plain
LinearRegression land at essentially identical accuracy (the selected
α≈0.032 is small, meaning little regularization was actually needed to
generalize to the 2025 season) - but their *coefficients* are not equally
trustworthy; see the collinearity section below for why we read Ridge's.

**Random Forest did not beat the linear models here**, which surprised me
going in - trees are usually assumed to beat linear models "for free" by
capturing non-linearities. Two likely reasons: (1) with `age_squared`
already added, the main non-linear relationship in this data is handled
explicitly, closing most of the gap a tree would otherwise win back, and
(2) even with 4,258 training rows and default hyperparameters, the forest
has no tuning to avoid overfitting to idiosyncratic training-season
players. This is a real, checkable result, not a hand-wave - see
`reports/metrics.json`.

**Widening the window from 4 seasons to 10 seasons actually lowered R²**
(0.62 -> 0.48) rather than raising it, which is the single most important
result of extending the scope and is worth understanding rather than
glossing over. More training rows should help a model generalize - and
it would, if the relationship between stats and value were stable over
time. It isn't: Premier League transfer valuations have grown
substantially since 2016-17 (broadcast deal growth, inflation, wealthier
ownership groups), and **no feature in this project encodes which year a
row is from**. A below-average 2025 midfielder and an above-average 2017
midfielder with matching stats get an identical prediction, even though
the real market would price them very differently. Over a short 4-season
window that effect is small enough to ignore; over a full decade it
becomes the dominant source of error, which is exactly why the naive
position-median baseline (blending medians across 10 seasons of a rising
market) got so much worse that it went negative. The honest fix would be
an explicit season/year feature or a year-indexed inflation adjustment on
the target - deliberately left out here so this result stays visible
rather than quietly regressed away.

## Charts (`reports/`)

- `predicted_vs_actual.png` - tight along y=x under ~EUR30M, but the model
  systematically **under-predicts** the handful of highest-value stars
  (e.g. Erling Haaland's actual EUR200M is predicted at ~EUR150M). This is
  the single clearest visual evidence of the model's biggest weakness - see
  below.
- `residuals_vs_predicted.png` - residual spread visibly widens as
  predicted value increases (heteroscedasticity): the model is precise for
  squad players, much less precise for stars.
- `ridge_coefficients.png` - `age_at_season_end` (+2.7) and `age_squared`
  (-3.2) are the two largest-magnitude coefficients, and their opposite
  signs are exactly the point: value rises with age then falls, a
  parabola no single linear age term could represent. `minutes_played` is
  the next strongest positive driver.
- `over_under_valued.png` - the 15 players the market pays most *above*
  what their raw stats predict (William Saliba, Declan Rice, Moises
  Caicedo, Bukayo Saka, Alexander Isak...) and the 15 it pays *below*
  (mostly squad/rotation players at mid-table clubs). This list is itself
  evidence for the limitations section: nearly every "overvalued" name is
  an elite, high-reputation, high-marketability player - exactly the
  intangibles a stats-only model cannot see.

## Honest limitations

**Where a linear model is the wrong tool.** Player value is not a smooth,
additive function of stats - it has thresholds and interactions a linear
model can't represent: a single wonder-goal against a rival can move a
player's valuation more than a season of solid-but-unspectacular numbers;
a young player's stats matter far more than an aging player's identical
stats because of resale/potential value; and a defensive midfielder's
value depends on stats we don't have (progressive passes, tackles won) far
more than goals/assists. A linear model also can't cap or floor a
prediction - it will happily extrapolate a nonsensical value for a stat
combination outside the training range. The RandomForest exists in this
project specifically to measure that cost, and here the cost turned out to
be small (R² 0.44 vs 0.48) - evidence the *available* features are mostly
linear-ish once age is fixed, not evidence that value itself is linear.

**Which features are collinear** (correlation matrix computed on the
actual feature set):
- `appearances` and `minutes_played`: r = 0.91 - nearly redundant; both
  measure "how much did this player play."
- `goals` and `goals_per_90`: r = 0.77; `assists` and `assists_per_90`:
  r = 0.70 - the per-90 rate is derived from the raw count and shares its
  numerator, so of course they move together.
- `age_at_season_end` and `age_squared`: r = 1.00 by construction - this
  one is *intentional* collinearity (a polynomial term), not accidental,
  and is exactly why Ridge's coefficients are the trustworthy ones (see
  below), not plain OLS's.
- `appearances`/`minutes_played` and `yellow_cards`: r ~0.54-0.57 - more
  minutes simply means more opportunities to be booked, not that playing
  time causes cards.

Under this much collinearity, plain `LinearRegression`'s individual
coefficients can be unstable (small data changes can swing or even flip
them) even though its *predictions* are fine - which is why
`ridge_coefficients.png` reads Ridge's shrunk coefficients rather than
OLS's for the "what drives value" chart.

**Why age is not linear, and the fix used.** Value rises through a
player's early-to-mid 20s, peaks, then declines - a parabola, not a slope.
A single linear `age` term forces one constant slope across a 17-year-old
prospect and a 34-year-old veteran, which can't represent a peak at all.
The fix used here is the standard one for representing non-monotonic
effects while staying inside a linear model: add `age_at_season_end ** 2`
as a second feature, so the model fits `b1*age + b2*age^2`, a curve
instead of a line. The fitted signs (`age`: +2.7, `age_squared`: -3.2)
confirm the expected rise-then-fall shape.

**What the model fundamentally cannot know**, and roughly how much of the
error that explains: contract length/expiry, agent negotiating power,
club wealth and wage structure, transfer hype/media narrative, injury
history, international reputation, and resale/potential value for young
players. The evidence for how much this matters is in the charts: for the
bulk of the test season (~EUR 30M and under) predictions track the y=x
line closely, meaning stats alone explain most of the variance for
ordinary squad players. Nearly all of the model's largest errors - Saliba,
Rice, Caicedo, Saka, Isak - are elite players whose value is driven by
reputation and marketability on top of, not instead of, good stats; the
model captures the "good stats" part but has no feature for the premium
the market adds on top. Given the overall MAE is ~EUR 12.8M but the worst
individual errors reach ~EUR 81M concentrated in a small number of star
players, it's fair to say intangibles this model can't see account for a
small fraction of *total* prediction count that's wrong but a large
fraction of the *largest* errors by euro amount - the model is good at
pricing depth players and bad at pricing stars, which is the opposite of
what a scout would find most useful.

**A decade also introduces a problem 4 seasons mostly hid: non-stationary
prices.** See the "widening the window" note above - none of these models
know what year a row is from, so a decade of transfer-market inflation
gets absorbed into the error term instead of being explained. This is
very likely the single largest driver of the R² drop from 0.62 (4 seasons)
to 0.48 (10 seasons), and it's a more fundamental limitation than any of
the feature-level issues above: it's not a missing feature so much as a
violated assumption (that the stats-to-value relationship is constant
over the training window), and it would keep getting worse the further
back the window is extended.

## Feature list

Counting stats (summed per player per season, Premier League matches
only): `appearances`, `minutes_played`, `goals`, `assists`,
`yellow_cards`, `red_cards`.

Engineered: `goals_per_90`, `assists_per_90` (control for playing time -
two players with 5 goals aren't equally productive if one played 900
minutes and the other 3,000), `age_at_season_end` (fractional years, not
integer - two players both "27" by whole-year age can be nearly a year
apart), `age_squared` (see above), `height_in_cm` (weak positional proxy;
kept as it's directly available and free), `position` (one-hot,
`drop="first"` to avoid the dummy-variable trap with the intercept).

Dropped from the original ask: **`games_started`** - `appearances.csv`
has no starting-lineup flag, only `minutes_played`. Faking a threshold
(e.g. "60+ minutes = start") would produce a feature that looks precise
but isn't; `minutes_played` and the per-90 rates already capture
playing-time signal without a fabricated proxy. Also dropped: a per-season
**club** feature - `players.csv` only records each player's *current*
club, not their club during a historical season, so using it would
silently mislabel past seasons.

## Project layout

```
config.py                  paths, seasons, thresholds - no magic numbers elsewhere
src/
  fetch.py                 download + cache raw CSVs
  build_dataset.py         season aggregation + leak-safe valuation join
  features.py              feature engineering, log1p target
  train.py                 3 models in sklearn Pipelines, evaluation
  predict.py               CLI lookup
  visualize.py             the 4 matplotlib charts
tests/
  test_build_dataset.py    season-end date + fractional age
  test_features.py         per-90 rates, age_squared, minute filtering, log target
data/raw/                  cached CSVs (gitignored)
data/processed/            player_seasons.csv
models/                    joblib-persisted pipelines (gitignored)
reports/                   metrics.json, test_predictions.csv, PNG charts
```
