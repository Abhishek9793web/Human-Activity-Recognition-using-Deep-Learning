import argparse
from data.loader import load_data
from models.lstm import build_lstm
from models.cnn1d import build_cnn1d
from models.cnnlstm import build_cnnlstm
from utils.train import train_model
from utils.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "cnn1d", "cnnlstm"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    (X_train, y_train), (X_test, y_test) = load_data()
    input_shape = (X_train.shape[1], X_train.shape[2])
    n_classes = len(set(y_train))

    if args.model == "lstm":
        model = build_lstm(input_shape, n_classes)
    elif args.model == "cnn1d":
        model = build_cnn1d(input_shape, n_classes)
    else:
        model = build_cnnlstm(input_shape, n_classes)

    model, history = train_model(model, X_train, y_train, X_test, y_test, epochs=args.epochs, batch_size=args.batch_size)

    target_names = [f"Class {i}" for i in range(1, n_classes+1)]
    evaluate_model(model, X_test, y_test, target_names)

if __name__ == "__main__":
    main()