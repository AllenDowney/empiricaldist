# Hazard Function (Hazard)

A `Hazard` object represents a mapping from a quantity, t, to a conditional probability: of all values in the distribution that are at least t, what fraction are exactly t?

`Hazard` is a subclass of a Pandas `Series`, so it has all `Series` methods, although some are overridden to change their behavior.

::: empiricaldist.Hazard
