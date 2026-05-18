# Telemetry-Driven R&D Test Minimization Engine for Formula 1 Development

This repository is a Formula 1 R&D decision-support prototype. It is not a
game setup tool. F1 2020 telemetry is supported as an optional validation proxy
because real F1 telemetry and R&D databases are private.

The workflow is:

```text
telemetry observation
-> symptom identification
-> probabilistic root-cause hypotheses
-> R&D test library query
-> expected information gain / cost optimization
-> recommended minimum-cost test sequence
-> belief update after test outcome
```

## Current Scope

The project now supports an expanded config-driven diagnostic taxonomy:

- 45 executable symptom definitions across entry, mid-corner, exit, high-speed,
  low-speed, braking, tire behavior, platform/aero, straight-line, and
  driver-input categories.
- 46 root-cause hypotheses across aero balance, platform aero, tires,
  mechanical setup, braking, differential/power delivery, straight-line power,
  driver input, and environment/data quality.
- 18 structured R&D tests with F1 interpretation, F1 2020 proxy, cost, outcomes,
  relevant hypotheses, and generated likelihood coverage.
- Generated `P(outcome | hypothesis, test)` tables from
  `config/likelihood_templates.json` with optional manual overrides.

The original `medium_speed_entry_to_apex_understeer` workflow is preserved as a
legacy-compatible symptom.

## Install

```bash
pip install -r requirements.txt
```

Streamlit is included for the optional dashboard. The core engine and CLI remain
CSV-first and do not require F1 2020 to be installed.

## CLI Usage

Generate synthetic telemetry for every supported symptom:

```bash
python -m src.cli generate-synthetic
```

Run a synthetic diagnosis:

```bash
python -m src.cli diagnose --scenario entry_understeer
```

Run a legacy hypothesis scenario:

```bash
python -m src.cli diagnose --scenario front_tire_thermal_saturation
```

Convert optional F1 2020 JSONL telemetry into the internal CSV schema:

```bash
python -m src.cli convert-f1-2020 --input logs/session.jsonl --output data/raw/session.csv
```

Diagnose any compatible CSV:

```bash
python -m src.cli diagnose-csv --file data/raw/session.csv
```

Run validation:

```bash
python -m src.cli validate --trials 100
```

Run the synthetic calibration report:

```bash
python -m src.cli calibrate --trials 100
```

The calibrated CLI default uses a low-value stopping floor of `0.15`, so the
optimizer stops before `max_tests` once the next test has little expected value.
Override it when you want a more exhaustive run:

```bash
python -m src.cli validate --trials 100 --min-expected-information-gain 0.0
```

Select optimizer modes:

```bash
python -m src.cli diagnose --scenario exit_oversteer --objective confidence_gain_per_cost
python -m src.cli diagnose --scenario high_speed_instability --two-step
```

Available optimizer objectives:

- `eig_per_cost`
- `eig`
- `confidence_gain_per_cost`
- `threshold_probability_per_cost`

## Streamlit Dashboard

Run:

```bash
streamlit run dashboard/app.py
```

The dashboard supports:

- synthetic scenario selection,
- CSV upload,
- detected symptoms and evidence,
- feature summary,
- belief distribution,
- ranked tests with optimizer explanation fields,
- manual outcome selection and belief update,
- validation snapshot,
- calibration sweep with symptom and hypothesis top-k metrics.

## F1 2020 Telemetry

The F1 2020 adapter expects JSONL records and maps common packet fields into the
internal schema:

- speed, steer, throttle, brake,
- lateral and longitudinal G,
- tire surface temperatures,
- tire wear,
- brake temperatures,
- wheel slip,
- gear, RPM, ERS deployment/energy,
- lap/session timing where available.

The engine intentionally remains CSV-based. There is no live UDP reader in this
version.

## Validation

Validation compares the optimizer against:

- random selection,
- cheapest-first,
- fixed sequence,
- grid/all-tests.

Reported metrics include:

- top-1 accuracy,
- top-3 containment,
- average tests used,
- average total cost,
- final confidence,
- cumulative top-3 confidence,
- confidence-threshold reach rate,
- average cost to correct diagnosis,
- per-hypothesis summaries in the returned validation object.

The calibration command also reports symptom confusion pairs, initial
true-hypothesis rank buckets, final optimizer top-1/top-3 behavior, and the
worst hypothesis confusions. It also includes likelihood diagnostics: highest
information-value tests, weakly separated hypothesis pairs, and test/outcome
coverage.

On the current synthetic benchmark:

- symptom detection reaches 36 / 45 top-1 and 43 / 45 top-3 containment,
- initial true-hypothesis top-3 containment is 0.889,
- optimizer final top-1 accuracy is about 0.680,
- optimizer final top-3 containment is about 0.930,
- average optimizer cost is about 6.51, below the cheapest-first baseline in
  the same 100-trial run.

This is a presentation-grade synthetic benchmark, not a claim of real F1
physical validity.

The expanded 46-hypothesis model is intentionally harder than the original v1
demo. Use validation as a comparative tool, not as proof of real F1 causality.

## Project Structure

```text
config/
  hypotheses.json
  symptoms.json
  symptom_hypothesis_map.json
  tests.json
  likelihood_templates.json
  likelihood_overrides.json
data/
  raw/
  synthetic/
  processed/
src/
  telemetry_loader.py
  synthetic_data.py
  feature_extractor.py
  symptom_identifier.py
  hypothesis_model.py
  test_library.py
  bayes.py
  optimizer.py
  outcome_classifier.py
  diagnosis_engine.py
  baselines.py
  validation.py
  f1_2020_adapter.py
  cli.py
dashboard/
  app.py
tests/
```

## Test

```bash
pytest
```

## Limitations

- Likelihoods are plausible template-generated priors, not learned from real R&D
  outcomes.
- Synthetic and F1 2020 telemetry validate workflow mechanics, not real-car
  physical causality.
- The optimizer includes one-step EIG and optional shallow two-step lookahead,
  not full POMDP planning.
- Detection rules are transparent heuristics; thresholds need calibration for
  any real telemetry source.
- No live F1 2020 UDP or real F1 database integration is included.

## Future Work

- Calibrate likelihoods from historical simulator/test outcomes.
- Add manual review tools for likelihood overrides.
- Add richer segment alignment and reference-lap comparison.
- Integrate CFD, simulator, wind-tunnel, and rig test databases.
- Add POMDP or Monte Carlo tree search for multi-step planning.
