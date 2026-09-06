# AI-QbD Design & Phase-Behavior Prediction of CUR–SAC/VAL Co-Crystals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/RDKit-2023.09+-green.svg)](https://www.rdkit.org/)
[![GROMACS](https://img.shields.io/badge/GROMACS-2024-red.svg)](https://www.gromacs.org/)

This repository contains the complete computational workflow, data pipelines, machine learning scripts, and statistical Quality by Design (QbD) optimization code for the manuscript:

> **Artificial Intelligence-Driven Design and Phase-Behavior Prediction of Novel Curcumin and Sacubitril/Valsartan Co-Crystals Using a Quality by Design Framework**

---

## Abstract Overview
Formulation of Sacubitril/Valsartan (SAC/VAL) multi-component solids poses structural and thermodynamic stability challenges. This study implements an integrated 4-stage *in silico* AI-QbD framework to design and predict the manufacturing design space for a novel Nutra-Pharma ternary co-crystal involving Curcumin (CUR), Sacubitril and Valsartan. 

The computational pipeline couples:
1. **Hansen Solubility Parameter (HSP)** thermodynamic screening.
2. **AutoDock Vina** virtual screening & multi-nanosecond **GROMACS Molecular Dynamics (MD)** simulations.
3. **Random Forest Classifier** trained on 500 pharmaceutical co-crystal systems for phase stability prediction.
4. **Central Composite Design (CCD)** Response Surface Methodology via Stat-Ease 360 to map the safe manufacturing operating window.

---

## Repository Structure

```text
.
├── main_pipeline.py      # Complete Python pipeline for HSP, ML, Docking & MD analysis
├── README.md             # Project description and setup instructions
└── LICENSE               # Open-source license

