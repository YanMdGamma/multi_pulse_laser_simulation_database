## Description

This repository contains all data files related to simulating multi-pulse femtosecond laser processing of diamond and the preparation of color centers using LAMMPS. It also includes Python code implementing the **DBSCAN** algorithm for **amorphous structure analysis**.

## Github_code

- **non-crystaline.py**: the core detection algorithm.

  Required input file:
   `filename = "./RMlloweng_other/RMlloweng_other_NV.txt"`
   The path can be either an absolute path or a relative path.

  You can choose to perform clustering on **N atoms**, **NV center structures**, or **C atoms**.
   The input file must be preprocessed into a **LAMMPS input data format without a header**.

  - `min_samples`: the minimum number of samples (points) required to form a cluster
  - `eps`: epsilon neighborhood radius

- The results are saved as images.

> [!IMPORTANT]
>
> The neighborhood radius should be selected properly to avoid cases where a cluster/group contains no atoms.

## Contributors

- Mengzhi Yan
- Fengzhou Fang
