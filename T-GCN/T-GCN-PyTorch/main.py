import argparse
import traceback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import utils.callbacks
import utils.data
import utils.email
import utils.logging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import time
import os
import csv
# print("CWD =", os.getcwd())

DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
    "dpos": {"feat": "data/dpos/dpos5.csv", "adj": "data/dpos/dpos_adj.csv"},
}


def get_model(args, dm):
    model = None
    if args.model_name == "GCN":
        model = models.GCN(adj=dm.adj, input_dim=args.seq_len, output_dim=args.hidden_dim)
    if args.model_name == "GRU":
        model = models.GRU(input_dim=dm.adj.shape[0], hidden_dim=args.hidden_dim)
    if args.model_name == "TGCN":
        model = models.TGCN(adj=dm.adj, hidden_dim=args.hidden_dim)
    return model


def get_task(args, model, dm):
    mean, sigma = dm.feat_max_val
    task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
        model=model, mean=mean, sigma=sigma, **vars(args)
    )
    return task


def get_callbacks(args):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="train_loss")
    plot_validation_predictions_callback = utils.callbacks.PlotValidationPredictionsCallback(monitor="train_loss")
    early_stop_callback = EarlyStopping(
        monitor="RMSE",   # 关键：要和你任务里 log 的名字一致
        mode="min",
        patience=30,
        verbose=True,
        check_on_train_epoch_end=False,  # 让它在 val 之后判断
    )
    callbacks = [
        checkpoint_callback,
        plot_validation_predictions_callback,
        early_stop_callback,
    ]
    return callbacks


def main_supervised(args):
    dm = utils.data.SpatioTemporalCSVDataModule(
        feat_path=DATA_PATHS[args.data]["feat"], adj_path=DATA_PATHS[args.data]["adj"], **vars(args)
    )
    model = get_model(args, dm)
    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    task = get_task(args, model, dm)
    callbacks = get_callbacks(args)
    # trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    feat_path = DATA_PATHS[args.data]["feat"]
    run_name = os.path.splitext(os.path.basename(feat_path))[0]
    logger = TensorBoardLogger("lightning_logs", name=f'{run_name}_{args.test_choose}')
    trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    accelerator=args.accelerator,
    devices=args.devices,
    callbacks=callbacks,
    logger = logger
    )
    t0 = time.perf_counter()
    trainer.fit(task, dm)
    t1 = time.perf_counter()
    train_seconds = t1 - t0
    append_params_time_csv(f"lightning_logs/{run_name}_{args.test_choose}/params_time.csv", total_params, trainable_params, train_seconds)
    results = trainer.validate(datamodule=dm)
    return results


def main(args):
    rank_zero_info(vars(args))
    results = globals()["main_" + args.settings](args)
    return results

def append_params_time_csv(csv_path, total_params, trainable_params, train_seconds):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["total_params", "trainable_params", "train_seconds"])
        writer.writerow([total_params, trainable_params, f"{train_seconds:.6f}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    pl.seed_everything(42, workers=True)
    parser.add_argument("--test_choose", type=int, help="振动段选择", choices=(0, 1), default=1)
    parser.add_argument(
        "--data", type=str, help="The name of the dataset", choices=("shenzhen", "losloop", "dpos"), default="dpos"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("GCN", "GRU", "TGCN"),
        default="TGCN",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised",),
        default="supervised",
    )
    parser.add_argument("--max_epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Device type: 'cpu', 'gpu', or 'auto'")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use. E.g., 1 for 1 GPU")
    parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")
    parser.add_argument("--send_email", "--email", action="store_true", help="Send email when finished")

    temp_args, _ = parser.parse_known_args()

    parser = getattr(utils.data, temp_args.settings.capitalize() + "DataModule").add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.capitalize() + "ForecastTask").add_task_specific_arguments(parser)

    args = parser.parse_args()
    utils.logging.format_logger(pl._logger)
    if args.log_path is not None:
        utils.logging.output_logger_to_file(pl._logger, args.log_path)

    try:
        results = main(args)
    except:  # noqa: E722
        traceback.print_exc()
        if args.send_email:
            tb = traceback.format_exc()
            subject = "[Email Bot][❌] " + "-".join([args.settings, args.model_name, args.data])
            utils.email.send_email(tb, subject)
        exit(-1)

    if args.send_email:
        subject = "[Email Bot][✅] " + "-".join([args.settings, args.model_name, args.data])
        utils.email.send_experiment_results_email(args, results, subject=subject)
