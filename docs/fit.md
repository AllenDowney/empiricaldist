# Fitting distributions

The `empiricaldist.fit` module fits parametric distributions to empirical data and compares them visually. Fitters minimize vertical CDF error on a grid of probabilities (CDF matching), not moment-matching alone.

See the [API module docstring](https://github.com/AllenDowney/empiricaldist/blob/master/empiricaldist/fit.py) for design notes on `soft_l1` loss and tail vs bulk fitting.

::: empiricaldist.fit
    options:
      show_submodules: false
      members_order: source
      filters:
        - "!^_"
