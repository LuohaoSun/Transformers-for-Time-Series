# use __main__ to prevent errors caused by multiprocessing.
import sys

sys.path.append("./")


if __name__ == "__main__":
    # Step 1. Choose a dataset (customized dataset from framework.forecasting.forecasting_datamodule)
    from utils.create_datamodule.forecasting_datamodule import ForecastingDataModule

    datamodule = ForecastingDataModule(
        csv_file_path="data/BJinflow/in_10min_trans.csv",
        stride=1,
        input_length=24,
        output_length=24,
        batch_size=128,
        train_val_test_split=(0.6, 0.2, 0.2),
        num_workers=7,
        normalization="01",
    )

    # Step 2. Choose a model (ForecastingFramework from models.forecasting_models)
    from backbones import LSTMBackbone

    backbone = LSTMBackbone(
        in_features=276,
        hidden_features=64,
        auto_recurrsive_steps=24,
        num_layers=1,
    )

    # Step 3. Choose a framework (ForecastingFramework from framework.forecasting_framework)
    from frameworks import ForecastingFramework

    framework = ForecastingFramework(
        backbone=backbone,
        backbone_out_features=64,
        out_seq_len=24,
        out_features=276,
        evry_n_epochs=10,
        fig_size=(10, 5),
    )

    # Step 4. fit and test
    framework.fit(datamodule,max_epochs=100)
    framework.test(datamodule)

    from utils.visualization import SeriesPlotter

    sample_data = datamodule.test_dataset[0]
    x = sample_data[0].unsqueeze(0)
    y = sample_data[1].unsqueeze(0)
    y_hat = framework(x)
    SeriesPlotter.plot_and_show({"y": y, "y_hat": y_hat})
