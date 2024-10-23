# ml_benchmark
Some custom cases or POC validation, that are not yet integrated in https://github.com/IntelPython/scikit-learn_bench 
Cases:
* Data management update benchmarks

## Prerequisites
TBD

## Running benchmark tests
```bash
pytest benchmarks/ --benchmark-json=results.json
```
Running tests and saving the current run into `STORAGE`, see [1]
```bash
pytest benchmarks/ --benchmark-json=results.json --benchmark-autosave
```

## Creating `.csv` report
```bash
pytest-benchmark compare results.json --csv=results.csv --group-by='name'
```

## Optional: creating histogram
Note: make sure that `pytest-benchmark[histogram]` installed
```bash
pip install pytest-benchmark[histogram]
pytest -vv benchmarks/ --benchmark-autosave --benchmark-histogram
pytest-benchmark compare .benchmarks/Linux-CPython-3.10-64bit/* --histogram
```

## Advanced running example
```
pytest benchmarks/ --benchmark-columns='min, max, mean, stddev, median, rounds, iterations' --benchmark-json=results.json --benchmark-autosave
pytest-benchmark compare results.json --csv=results.csv --group-by='name'
```


[1] https://pytest-benchmark.readthedocs.io/en/latest/usage.html