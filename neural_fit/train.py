"""Train a 2-10-1 neural network to approximate z = x**2 + y**2 + 10."""

import argparse
import copy
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def build_model():
    # Three layers when counting input, hidden, and output layers; 41 parameters.
    return nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 1)).double()


def target(xy):
    return (xy**2).sum(dim=1, keepdim=True) + 10.0


def predict(model, xy, radius, target_mean, target_std):
    with torch.no_grad():
        return model(xy / radius) * target_std + target_mean


def save_plots(model, radius, mean, std, history, output):
    axis = np.linspace(-radius, radius, 101)
    xx, yy = np.meshgrid(axis, axis)
    xy = torch.from_numpy(np.column_stack([xx.ravel(), yy.ravel()]))
    truth = xx**2 + yy**2 + 10
    fitted = predict(model, xy, radius, mean, std).numpy().reshape(xx.shape)
    error = fitted - truth
    fig = plt.figure(figsize=(12, 9), layout="constrained")
    for index, (values, title) in enumerate(
        [(truth, "True: z = x² + y² + 10"), (fitted, "Neural network: 2 → 10 → 1")], 1
    ):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        ax.plot_surface(xx, yy, values, cmap="viridis", linewidth=0)
        ax.set(xlabel="x", ylabel="y", zlabel="z", title=title, zlim=(10, 10 + 2 * radius**2))
    ax = fig.add_subplot(2, 2, 3)
    limit = max(float(np.abs(error).max()), 1e-12)
    im = ax.imshow(error, origin="lower", extent=[-radius, radius] * 2,
                   cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set(xlabel="x", ylabel="y", title="Prediction error: predicted − true")
    fig.colorbar(im, ax=ax, label="z error")
    ax = fig.add_subplot(2, 2, 4)
    steps, train_mse, validation_mse = np.array(history).T
    ax.semilogy(steps, train_mse, label="Training")
    ax.semilogy(steps, validation_mse, label="Validation")
    ax.set(xlabel="Optimizer step (Adam, then L-BFGS)", ylabel="MSE in original z units",
           title="Learning curve")
    ax.legend()
    fig.savefig(output / "fit.png", dpi=160)
    plt.close(fig)


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = build_model()

    # Independent random samples; the test set is never used during training.
    samples = [torch.rand(n, 2, dtype=torch.float64) * (2 * args.radius) - args.radius
               for n in (4096, 1024, 2048)]
    train_xy, val_xy, test_xy = samples
    train_z, val_z, test_z = [target(xy) for xy in samples]
    mean, std = float(train_z.mean()), float(train_z.std())
    train_input = train_xy / args.radius
    train_target = (train_z - mean) / std
    loss_fn = nn.MSELoss()
    history = []
    best = {"validation_mse": float("inf"), "state": None, "step": 0}

    def record(step):
        with torch.no_grad():
            train_mse = float(loss_fn(predict(model, train_xy, args.radius, mean, std), train_z))
            val_mse = float(loss_fn(predict(model, val_xy, args.radius, mean, std), val_z))
        history.append([step, train_mse, val_mse])
        if val_mse < best["validation_mse"]:
            best.update(validation_mse=val_mse, state=copy.deepcopy(model.state_dict()), step=step)
        return train_mse, val_mse

    record(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for step in range(1, args.epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn(model(train_input), train_target)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.epochs:
            train_mse, val_mse = record(step)
            if step % 500 == 0 or step == args.epochs:
                print(f"Adam {step:4d}: train MSE={train_mse:.6f}, validation MSE={val_mse:.6f}", flush=True)

    # L-BFGS is effective for polishing this small, smooth regression model.
    model.load_state_dict(best["state"])
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                                  tolerance_grad=1e-10, tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(model(train_input), train_target)
        loss.backward()
        return loss

    for round_index in range(1, 41):
        optimizer.step(closure)
        # This axis counts optimizer calls, not internal L-BFGS iterations.
        train_mse, val_mse = record(args.epochs + round_index)
        if round_index % 10 == 0:
            print(f"L-BFGS {round_index:2d}: train MSE={train_mse:.6f}, validation MSE={val_mse:.6f}", flush=True)

    model.load_state_dict(best["state"])
    model.eval()
    predicted = predict(model, test_xy, args.radius, mean, std)
    residual = predicted - test_z
    test_mse = float((residual**2).mean())
    examples_xy = torch.tensor([[0, 0], [1, 2], [3, 4], [-2, 1], [5, 5]], dtype=torch.float64)
    examples_pred = predict(model, examples_xy, args.radius, mean, std).flatten().tolist()
    examples_true = target(examples_xy).flatten().tolist()
    examples = [dict(x=float(x), y=float(y), true=z, predicted=p)
                for (x, y), z, p in zip(examples_xy.tolist(), examples_true, examples_pred)]
    metrics = {
        "function": "z = x^2 + y^2 + 10", "architecture": [2, 10, 1],
        "activation": "Tanh", "parameter_count": sum(p.numel() for p in model.parameters()),
        "seed": args.seed, "range": [-args.radius, args.radius],
        "sample_counts": {"train": 4096, "validation": 1024, "test": 2048},
        "adam_steps": args.epochs, "lbfgs_calls": 40,
        "selected_step": best["step"], "validation_mse": best["validation_mse"],
        "test_mse": test_mse, "test_rmse": test_mse**0.5,
        "test_mae": float(residual.abs().mean()),
        "test_max_absolute_error": float(residual.abs().max()),
        "test_r2": 1 - float((residual**2).sum() / ((test_z - test_z.mean())**2).sum()),
        "examples": examples,
    }
    torch.save({"state_dict": model.state_dict(), "radius": args.radius,
                "target_mean": mean, "target_std": std, "architecture": [2, 10, 1],
                "activation": "Tanh"}, output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["x", "y", "true_z", "predicted_z", "error"])
        writer.writerows(np.column_stack([test_xy.numpy(), test_z.numpy(), predicted.numpy(), residual.numpy()]))
    with (output / "loss.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["optimizer_step", "train_mse", "validation_mse"])
        writer.writerows(history)
    save_plots(model, args.radius, mean, std, history, output)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved model, metrics, CSV files, and fit.png in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2000, help="Adam optimization steps")
    parser.add_argument("--radius", type=float, default=5.0, help="Fit on [-radius, radius] for both inputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--predict", type=float, nargs=2, metavar=("X", "Y"), help="Load saved model and predict")
    args = parser.parse_args()
    if args.epochs < 1 or not np.isfinite(args.radius) or args.radius <= 0:
        parser.error("epochs must be positive and radius must be finite and positive")
    if args.predict is None:
        train(args)
    else:
        checkpoint = torch.load(args.output / "model.pt", map_location="cpu", weights_only=True)
        model = build_model()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        xy = torch.tensor([args.predict], dtype=torch.float64)
        result = predict(model, xy, checkpoint["radius"], checkpoint["target_mean"], checkpoint["target_std"])
        print(f"x={args.predict[0]}, y={args.predict[1]}, predicted z={result.item():.6f}, true z={target(xy).item():.6f}")
        if max(abs(value) for value in args.predict) > checkpoint["radius"]:
            print("This point is outside the training range; extrapolation accuracy is not guaranteed.")


if __name__ == "__main__":
    main()
