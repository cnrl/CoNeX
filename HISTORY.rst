=======
History
=======

Unreleased
----------

* Add CI (lint, test matrix, e2e, wheel build) and a tag-driven release workflow.
* Add end-to-end tests for neurons, synapses, layers, networks, HTM, predictive coding, grid cells, voting and structures.
* Require pymonntorch 0.1.4 and declare sympy, which pymonntorch imports.
* Fix OutputLayer construction with the current pymonntorch signature.
* Fix save_ports for a port without a behavior.
* Fix sparse weight initialization on torch 1.x so the Sparse STDP rules run.
* Make SpatialPooler winner selection deterministic: ties go to the lowest column index.
