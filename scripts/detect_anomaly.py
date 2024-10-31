import sys
import torch
import argparse

sys.path.append("./")


def read_data(path):
    from t4ts.utils.data import SlidingWindowDataset

    x = SlidingWindowDataset.from_csv(path, window_size=512, stride=512)[0]
    x = (x - x.mean()) / x.std()
    # x = x / x.max()
    x = x.unsqueeze(0)
    return x


def compute_anomaly_score(x_hat: torch.Tensor, x: torch.Tensor):
    """
    x (Tensor): shape (batch_size, seq_len, n_features)
    Returns: Tensor of shape (batch_size, seq_len, 1) with the anomaly score for each step
    """
    import torch.nn.functional as F

    # anomaly_score = F.mse_loss(x_hat, x, reduction="none")
    anomaly_score = (x - x_hat).abs()
    anomaly_score = anomaly_score.mean(dim=-1, keepdim=True)
    anomaly_score = (F.sigmoid(anomaly_score) - 0.5) * 2
    return anomaly_score


def main(csv_path):
    from t4ts.pretrained.moment import MOMENT
    from t4ts.utils.visualization import SeriesPlotter

    x = read_data(csv_path)

    backbone = MOMENT(task="reconstruction")

    x_hat = backbone(x)
    anomaly_score = compute_anomaly_score(x_hat, x)

    series = {
        "original": x,
        "reconstructed": x_hat,
        "anomaly_score": anomaly_score,
    }
    SeriesPlotter.plot_and_show(series)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Detection Script")
    parser.add_argument("--path", type=str, help="Path to the CSV file")
    args = parser.parse_args()
    path = args.path or "data/bearing_fault_prediction/0/999.txt"

    main(path)
