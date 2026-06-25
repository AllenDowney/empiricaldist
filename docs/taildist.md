# Tail Distribution (TailDist)

A `TailDist` represents the tail distribution **P(X ≥ x)**. It is similar to a survival function, but a `Surv` object represents **P(X > x)**. The difference matters when a distribution has point masses, as empirical distributions often do.

See the [tail demo notebook](https://allendowney.github.io/empiricaldist/tail_demo.html) for examples.

::: empiricaldist.TailDist
