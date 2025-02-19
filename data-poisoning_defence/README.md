# README.md

# Data Poisoning Defense

This project implements various defenses against data poisoning attacks in machine learning models, specifically focusing on the use of forest-based methods.

## Overview

Data poisoning is a type of attack where an adversary manipulates the training data to degrade the performance of a machine learning model. This project aims to provide robust defenses against such attacks by utilizing advanced training techniques and optimizers.

## Features

- Implementation of various training strategies to mitigate data poisoning.
- Support for different optimizers including SGD, Adam, and their variants.
- Integration of learning rate schedulers, including Cosine Annealing Scheduler for dynamic learning rate adjustment.

## Requirements

To run this project, you need to install the required Python packages. You can do this by running:

```
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
   ```
   git clone <repository-url>
   cd data-poisoning_defence
   ```

2. Prepare your dataset and configure the training parameters in `forest/victims/training.py`.

3. Run the training script:
   ```
   python forest/victims/training.py
   ```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.