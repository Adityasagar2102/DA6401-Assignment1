import os
import json
import numpy as np
from argparse import Namespace
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork

# Import the shared parser and helper from train.py
from train import build_parser, parse_hidden_size


def load_model(model_path):
    """
    Load trained model from disk.
    Exact pattern from updated assignment instructions:

        data = np.load(model_path, allow_pickle=True).item()
        return data
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def get_config(model_path):
    """Locate best_config.json next to the model file."""
    model_dir = os.path.dirname(os.path.abspath(model_path))
    config_path = os.path.join(model_dir, "best_config.json")

    if not os.path.exists(config_path):
        # fallback: look in same dir as inference.py
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    config["hidden_size"] = parse_hidden_size(config["hidden_size"])
    return config


def evaluate_model(model, X_test, y_test):
    """Run forward pass and compute all metrics."""
    # Model returns raw LOGITS (not softmax) per updated spec
    logits = model.forward(X_test)

    predictions = np.argmax(logits, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    accuracy  = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="macro", zero_division=0)
    recall    = recall_score(true_labels, predictions, average="macro", zero_division=0)
    f1        = f1_score(true_labels, predictions, average="macro", zero_division=0)

    loss = model.loss.forward(logits, y_test)

    return {
        "logits":    logits,
        "loss":      loss,
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1
    }


def main():
    # Same CLI as train.py per updated instructions
    parser = build_parser()
    args = parser.parse_args()

    args.hidden_size = parse_hidden_size(args.hidden_size)

    # Determine model path
    if args.model_path is None:
        # Default: look for best_model.npy next to this script
        args.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "best_model.npy"
        )

    # Load config from json (overrides CLI defaults with saved best config)
    config = get_config(args.model_path)
    cli = Namespace(**config)

    # Build model and load weights using the exact pattern from updated spec
    model = NeuralNetwork(cli)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    # Load test data
    _, _, _, _, X_test, y_test = load_data(args.dataset)

    results = evaluate_model(model, X_test, y_test)

    print("Evaluation Results")
    print("------------------")
    print(f"Loss:      {results['loss']:.4f}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")

    return results


if __name__ == '__main__':
    main()