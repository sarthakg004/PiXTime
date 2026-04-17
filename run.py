import argparse
import math
import os
import torch
import random
import copy

import numpy as np
import torch.nn as nn

from datetime import datetime

from dataset.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, clever_format
from utils.metrics import metric

import models.DLinear as DLinear
import models.PatchTST as PatchTST
import models.iTransformer as iTransformer
import models.TimeXer as TimeXer
import models.PiXTime as PiXTime


def get_device():
    """Safe device selection for Mac (CPU / MPS / CUDA)"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def work_process(args):
    args.device = get_device()

    train_loader, enc_in = data_provider(args, flag='train')
    val_loader, _ = data_provider(args, flag='val')
    test_loader, _ = data_provider(args, flag='test')

    # Model selection
    if args.model == 'DLinear':
        model = DLinear.Model(seq_len=args.seq_len, pred_len=args.pred_len, enc_in=enc_in)
    elif args.model == 'PatchTST':
        model = PatchTST.Model(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            patch_len=args.patch_len,
            d_model=args.d_model,
            dropout=args.dropout,
            factor=args.factor,
            n_heads=args.n_heads,
            d_ff=args.en_d_ff,
            e_layers=args.en_layers,
            enc_in=enc_in
        )
    elif args.model == 'iTransformer':
        model = iTransformer.Model(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            d_model=args.d_model,
            dropout=args.dropout,
            factor=args.factor,
            n_heads=args.n_heads,
            d_ff=args.en_d_ff,
            e_layers=args.en_layers,
            enc_in=enc_in
        )
    elif args.model == 'TimeXer':
        model = TimeXer.Model(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            patch_len=args.patch_len,
            enc_in=enc_in,
            d_model=args.d_model,
            dropout=args.dropout,
            factor=args.factor,
            n_heads=args.n_heads,
            d_ff=args.en_d_ff,
            e_layers=args.en_layers
        )
    elif args.model == 'PiXTime':
        model = PiXTime.Model(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            patch_len=args.patch_len,
            n_vars=enc_in,
            d_model=args.d_model,
            dropout=args.dropout,
            factor=args.factor,
            n_heads=args.n_heads,
            en_d_ff=args.en_d_ff,
            de_d_ff=args.de_d_ff,
            en_layers=args.en_layers,
            de_layers=args.de_layers
        )
    else:
        raise ValueError("Unknown model")

    model = model.to(args.device)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # ================= TRAIN =================
    for epoch in range(args.train_epochs):
        train_loss = []
        model.train()

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            optimizer.zero_grad()

            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            batch_x_mark = batch_x_mark.float().to(args.device)
            batch_y_mark = batch_y_mark.float().to(args.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(args.device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            loss = loss_func(outputs, batch_y)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_loss)

        # ================= VALIDATION =================
        val_loss = []
        model.eval()

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x = batch_x.float().to(args.device)
                batch_y = batch_y.float().to(args.device)
                batch_x_mark = batch_x_mark.float().to(args.device)
                batch_y_mark = batch_y_mark.float().to(args.device)

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(args.device)

                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]

                loss = loss_func(outputs, batch_y)
                val_loss.append(loss.item())

        val_loss = np.mean(val_loss)

        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}")
        adjust_learning_rate(optimizer, epoch + 1, args)

    # ================= TEST =================
    preds, trues = [], []
    model.eval()

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            batch_x_mark = batch_x_mark.float().to(args.device)
            batch_y_mark = batch_y_mark.float().to(args.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(args.device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]

            preds.append(outputs.cpu().numpy())
            trues.append(batch_y[:, -args.pred_len:, f_dim:].cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print(f"Test -> MSE: {mse}, MAE: {mae}")

    # Save results
    with open(args.evaluation, 'a') as f:
        f.write(
            f"data={args.data}, model={args.model}, "
            f"seq_len={args.seq_len}, pred_len={args.pred_len}, "
            f"features={args.features}, d_model={args.d_model}, "
            f"MSE={mse}, MAE={mae}\n"
        )


def run():
    parser = argparse.ArgumentParser(description='Time Series Forecasting')

    parser.add_argument('--model', type=str, default='PiXTime')
    parser.add_argument('--evaluation', type=str, default='./evaluation/')
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='dataset/ETT-small/')

    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--enc_in', type=int, default=7)

    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--en_d_ff', type=int, default=2048)
    parser.add_argument('--de_d_ff', type=int, default=2048)
    parser.add_argument('--en_layers', type=int, default=2)
    parser.add_argument('--de_layers', type=int, default=2)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')

    args = parser.parse_args()

    if args.seq_len % args.patch_len != 0:
        raise ValueError("seq_len must be divisible by patch_len")

    # Create model directory
    model_dir = os.path.join(args.evaluation, args.model)
    os.makedirs(model_dir, exist_ok=True)

    # Unique experiment filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (
        f"{args.data}_sl{args.seq_len}_pl{args.pred_len}"
        f"_ft{args.features}_dm{args.d_model}_{timestamp}"
    )

    args.evaluation = os.path.join(model_dir, f"{exp_name}.txt")

    print(args)
    work_process(args)

    # Safe cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    fix_seed = 47
    torch.set_num_threads(6)
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    run()
