# AI-QbD Design & Phase-Behavior Prediction of CUR–SAC/VAL Co-Crystals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/RDKit-2023.09+-green.svg)](https://www.rdkit.org/)
[![GROMACS](https://img.shields.io/badge/GROMACS-2024-red.svg)](https://www.gromacs.org/)

This repository contains the complete computational workflow, data pipelines, machine learning scripts, and statistical Quality by Design (QbD) optimization code supporting the research manuscript:

> **"Artificial Intelligence-Driven Design and Phase-Behavior Prediction of Curcumin and Sacubitril/Valsartan Co-Crystals Using a Quality by Design Framework"**

---

## 🔬 Abstract Overview
Formulating Sacubitril/Valsartan (SAC/VAL) as a stable multi-component solid faces significant thermodynamic and structural challenges. This study implements an integrated 4-stage *in silico* AI-QbD framework to design and map the manufacturing design space for a novel ternary "nutra-pharma" co-crystal comprising Curcumin (CUR), Sacubitril, and Valsartan.

The computational pipeline couples:
1. **Hansen Solubility Parameter (HSP)** thermodynamic miscibility screening.
2. **AutoDock Vina** blind docking & 100 ns **GROMACS Molecular Dynamics (MD)** simulations for atomic-level stability analysis.
3. **Random Forest Classifier** trained on 500 validated pharmaceutical co-crystal systems from the CSD database for crystalline-phase stability prediction.
4. **Central Composite Design (CCD)** Response Surface Methodology (RSM) via Stat-Ease 360 linking Critical Process Parameters (CPPs) to Critical Quality Attributes (CQAs).

---

## 🛠️ Repository Structure

```text
.
├── main_pipeline.py      # Integrated Python script for RDKit descriptors, HSP, ML, and MD analysis
├── README.md             # Project documentation and setup instructions
└── LICENSE               # Open-source MIT license
