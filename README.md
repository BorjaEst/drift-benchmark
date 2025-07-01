# drift-benchmark

Drift benchmark tool to test different benchmark frameworks

## Notes:

- Method: is a statistical aproach to evaluate statistics from data
- Implementation: is the way to apply a method to detect a drift
- Detector: is the specific implementation of a method.

## FAQ:

- Why to use "Covariate" drift instead of data-drift?
  From the literature, data drift is some times refered to any type of drift. Therefore by using "Covariate" is simpled to classify a sub-family where changes only occur on P(X)
