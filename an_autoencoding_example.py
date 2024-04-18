# Description: An example of how to use the AutoEncodingFramework

def main():
    import torch

    # Step 1. Choose a dataset (L.LightningDataModule)
    from data.bearing_fault_prediction.raw.fault_prediction_datamodule import FaultPredictionDataModule
    datamodule = FaultPredictionDataModule(batch_size=40)
    in_seq_len = datamodule.shape[1]    # 4096

    # Step 2. Choose a model (AutoEncodingFramework from models.autoencoding_models)
    from Modules.models.autoencoding_models import AutoEncoder
    model = AutoEncoder(
        encoder_in_seq_len=in_seq_len,
        encoder_hidden_len=(2048, 1024, 512, 256),
        encoder_out_seq_len=64,
        activation='gelu',

        lr=1e-3,
        max_epochs=100,
    )

    # Step 3. model trains itself with the datamodule
    model.fit(datamodule)
    model.test(datamodule)

    # Step 4. model predicts
    sample_data = torch.randn(4, 4096, 1)

    encoded = model.encode(sample_data)  # same as model.backbone(sample_data)
    decoded = model.decode(encoded)      # same as model.head(encoded)


if __name__ == '__main__':
    main()
