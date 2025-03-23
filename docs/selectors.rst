Selectors
=========

Selectors are the basic way to interact with a signal.  They can be used to "crop" a dataset or focus some
function.  In terms of `pyxem`/ `hyperspy` a selector is used anytime that a roi or a mask might be passed.

For example let's say we have a 4D dataset and we want to center the direct beam.  In pyxem we can do this by
calling the `center_direct_beam` method on the dataset which takes an optional ROI. Here the function
`center_direct_beam` is tied to the selector.