from rich import print
from rich.table import Table


def print_dict(data, header: tuple[str, str] = ("Key", "Value")):
    """
    使用 rich 库打印字典数据为表格形式。

    参数:
    data (dict): 要打印的字典数据。
    """

    table = Table()
    table.add_column(header[0])
    table.add_column(header[1])

    for key, value in data.items():
        table.add_row(key, str(value))

    print(table)


if __name__ == "__main__":
    # 示例字典
    hyperparameters = {
        "out_seq_len": 1,
        "num_classes": 4,
        "in_seq_len": 4096,
        "in_features": 1,
        "hidden_features": 128,
        "res_block_features": 512,
        "num_res_blocks": 10,
        "activation": "relu",
        "batch_size": 40,
        "transforms": None,
        "pin_memory": False,
        "lr": 0.001,
        "max_epochs": 100,
        "max_steps": -1,
    }

    # 调用函数打印表格
    print_dict(hyperparameters, header=("Hyperparameter", "Value"))
