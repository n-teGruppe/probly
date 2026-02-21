.. _installation:

The ``probly`` Python Package
=============================================
probly is a Python package for **uncertainty representation** and **quantification** for machine learning.

Installation
--------------

`probly` is intended to work with **Python 3.12.3 and above**. Installation can be done via `pip` and
or `uv`:

.. code-block:: sh

   pip install probly

or

.. code-block:: sh

   uv add probly

Quickstart
----------

`probly` makes it simple to make neural network models uncertainty-aware.
The library provides high-level wrappers for uncertainty representations
(e.g., Dropout, Ensembles, Bayesian NNs) and utilities for quantifying and
evaluating uncertainty.

Below is a minimal end-to-end example showing:

1. wrapping an existing neural network with an uncertainty representation,
2. training the model normally,
3. computing epistemic uncertainty,
4. performing an out-of-distribution (OOD) detection task.

.. code-block:: python

   import numpy as np
   import probly
   from probly.evaluation.tasks import out_of_distribution_detection
   from probly.representation.sampling import sampler_factory

   # 1. Create your neural network as usual
   net = ...  # define or load your PyTorch model

   # 2. Transform the model to enable MC Dropout-style sampling
   model = probly.transformation.dropout(net, p=0.5)

   # 3. Train the model normally using your existing training loop
   train(model)

   # 4. Sample predictions for in-distribution data
   data = ...  # load or create input batch
   sampler = sampler_factory(model, num_samples=20)
   preds = np.stack([p.detach().cpu().numpy() for p in sampler(data)], axis=1)

   # Convert representation to an epistemic uncertainty value
   eu = probly.quantification.classification.mutual_information(preds)

   # 5. Detect out-of-distribution samples
   data_ood = ...  # load or create OOD samples
   preds_ood = np.stack([p.detach().cpu().numpy() for p in sampler(data_ood)], axis=1)
   eu_ood = probly.quantification.classification.mutual_information(preds_ood)

   # Evaluate OOD detection using AUROC
   auroc = out_of_distribution_detection(eu, eu_ood)
   print("AUROC:", auroc)

The `quickstart` example uses Dropout as the uncertainty representation,
but `probly` also supports ensembles, evidential networks, Bayesian neural
networks, and various calibration methods. Any of these can be plugged into
the same workflow with minimal changes.
