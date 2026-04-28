from dataset.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader, Subset

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'electricity': Dataset_Custom,
    'exchange_rate': Dataset_Custom,
    'traffic': Dataset_Custom,
    'weather': Dataset_Custom
}


def data_provider(args, flag): #[batch size, length, n_vars]
    Data = data_dict[args.data]
    timeenc = 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = 'h'

    data_set = Data(
        args = args,
        root_path=args.root_path,
        data_path=args.data + '.csv',
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns='Monthly'
    )
    enc_in = data_set.enc_in

    if flag == 'train' and getattr(args, 'train_data_ratio', 1.0) < 1.0:
        ratio = float(args.train_data_ratio)
        if ratio <= 0:
            raise ValueError('train_data_ratio must be greater than 0')
        subset_len = max(1, int(len(data_set) * ratio))
        subset_indices = list(range(subset_len))
        data_set = Subset(data_set, subset_indices)
        print(f"train subset ratio={ratio:.3f}, subset_len={subset_len}")

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)

    return data_loader, enc_in
