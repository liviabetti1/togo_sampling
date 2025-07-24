
## Overview

The workflow is divided into two main steps:

### Step 1: Generate Costs

Sampling costs represent the difficulty or expense of collecting samples from different locations. This repo supports two ways to generate costs:

1. **Convenience-based Costs**

   - Costs are computed based on the distance from Lomé
   - The notebook [`costs/togo_generate_costs.ipynb`](costs/togo_generate_costs.ipynb) contains the code to generate and save these costs.
   - You can modify or extend this notebook to change the cost calculation methodology (add more cities,...)

2. **Cluster-based Sampling Costs**

   - Costs can also be generated or modeled based on clustering units or regions.

---

### Step 2: Generate Group Assignments

Group assignments are used to quantify the representativeness of a sample.

- Run the notebook [`groups/togo_generate_groups.ipynb`](groups/togo_generate_groups.ipynb) to generate and save group assignment dictionaries.
- Currently, group assignments that are used are regions. 

---

### Step 3: Run Sampling Experiments

- Sampling experiments are organized and executed via [`sampling/tools/togo_empty_initial_set_experiments.ipynb`](sampling/tools/togo_empty_initial_set_experiments.ipynb).
- Use this notebook to:
  - Configure experiment parameters 
  - Run multiple sampling strategies and budgets.
  - Results will be stored in the `output/` folder.

---

## Additional Notes

- The dataset paths and configurations are controlled via configuration files

---

## Getting Started

1. Run `costs/togo_generate_costs.ipynb` to generate sampling costs.
2. Configure parameters and run `sampling/tools/togo_empty_initial_set_experiments.ipynb` for your sampling experiments.
3. See logs, outputs, and csv files stored in the configured output directories.

---
