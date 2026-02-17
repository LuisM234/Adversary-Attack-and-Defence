# Adversarial Attack & Defence on Fashion-MNIST (PyTorch)

This repository contains my submission for a university adversarial machine learning competition.
The project implements **strong adversarial attacks (PGD)** and **robust adversarial training** for a neural network trained on the **Fashion-MNIST** dataset.

## 🏆 Competition Results

* **Adversarial Attack:** **1st Place**

  * Score: **100.00** (normalised distribution)
* **Adversarial Defence:** 66.06
* **Overall Score:** **83 / 100**

The goal was to build:

* A **strong attack** that reliably breaks models
* A **robust defence** that maintains accuracy under adversarial perturbations

---

## Overview

This project trains a fully connected neural network on Fashion-MNIST and evaluates its robustness against **Projected Gradient Descent (PGD)** attacks.

Key features:

* Iterative **PGD adversarial attack**
* **Adversarial training** with progressively strong perturbations
* Evaluation on **clean** and **adversarial** data
* Measurement of perturbation magnitude (L∞ distance)
* GPU support (if available)

---

## Model Architecture

A fully connected neural network:

```
Input (28×28) → Flatten
→ Linear(784 → 128) + ReLU
→ Linear(128 → 64) + ReLU
→ Linear(64 → 32) + ReLU
→ Linear(32 → 10)
→ LogSoftmax
```

---

## Adversarial Methodology

### Attack: Projected Gradient Descent (PGD)

* Norm: **L∞**
* Epsilon: up to **0.2**
* Step size (α): **0.01**
* Iterations: up to **40**
* Projection back to valid image range `[0, 1]`

This iterative attack is significantly stronger than single-step FGSM.

---

### Defence: Adversarial Training

Training uses a **mixed strategy**:

* Every 5th batch: **clean training**
* Remaining batches: **PGD adversarial examples**
* Strong training perturbation: ε = 0.20, 40 iterations

This improves robustness while maintaining reasonable clean accuracy.

---

## Evaluation

Two metrics are reported each epoch:

| Metric               | Description                     |
| -------------------- | ------------------------------- |
| Clean Accuracy       | Performance on normal test data |
| Adversarial Accuracy | Performance under PGD attack    |

Additional:

* Estimated **attack ability**
* Estimated **defence ability**
* Maximum L∞ perturbation (`p_distance`)

---

## Installation

### Requirements

* Python 3.8+
* PyTorch
* torchvision
* numpy
* pandas

Install dependencies:

```bash
pip install torch torchvision numpy pandas
```

---

## Usage

Train the model:

```bash
python main.py
```

Default parameters:

* Epochs: 20
* Batch size: 128
* Learning rate: 0.003

The model will:

* Download Fashion-MNIST automatically
* Train with adversarial training
* Evaluate after each epoch
* Save weights as:

```
<student_id>.pt
```

> Note: The student ID has been redacted for privacy.

---

## File Structure

```
.
├── main.py              # Training, attack, defence, evaluation
├── data/                # Automatically downloaded dataset
└── README.md
```

---

## Key Functions

| Function             | Purpose                            |
| -------------------- | ---------------------------------- |
| `adv_attack()`       | PGD attack for evaluation          |
| `adv_attack_train()` | Faster PGD for training            |
| `train()`            | Mixed clean + adversarial training |
| `eval_test()`        | Clean accuracy                     |
| `eval_adv_test()`    | Robustness evaluation              |
| `p_distance()`       | Maximum L∞ perturbation            |

---

## Results Summary

* Strong iterative PGD enabled **top attack performance**
* Mixed adversarial training improved robustness
* Demonstrates the **attack–defence trade-off** in adversarial ML

---

## Notes

* Designed for educational and research purposes
* Student identifier removed for privacy
* Competition-specific evaluation metrics may differ from standard benchmarks
* This README.md was made with the assistance of an AI tool

---

## Future Improvements

* Convolutional architecture (CNN)
* TRADES or advanced robust training methods
* Adaptive or targeted attacks
* Robustness vs. accuracy trade-off analysis
* AutoAttack evaluation

---

## License

MIT License (or add your preferred license)

---

## Author

Student competition submission – Adversarial Machine Learning
If you found this useful, feel free to ⭐ the repository!
