import os
from datetime import datetime
from Preproc.utilities import rawData_collate, sliding_window_data, data_split, MyDataset
from Preproc.loader import DataGenerator
from model import get_model
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--trial_name', type=str)
parser.add_argument('--data_loc', type=str)
parser.add_argument('--node_names', type=str)
parser.add_argument('--window_size', type=int, default=5)
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--train_split', type=float, default=0.6)
parser.add_argument('--valid_split', type=float, default=0.2)
parser.add_argument('--test_split', type=float, default=0.2)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dropout_rate', type=float, default=0.5)


args = parser.parse_args()
print(f'Training configs: {args}')
result_file = os.path.join('output', args.trial_name)
if not os.path.exists(result_file):
    os.makedirs(result_file)

node_names = pd.read_csv(args.node_names).to_numpy()

#preparing the input data
split = [args.train_split, args.test_split, args.valid_split]
rawData_arr = rawData_collate(args.data_loc, node_names)
s_arr, label_arr = sliding_window_data(rawData_arr, args.window_size)
x_list, label_list, normalize_statistics = data_split(s_arr, label_arr, split)

#saving normalize statistics
stat_file = os.path.join(result_file, 'normalize_statistics.npy')
with open(stat_file, 'wb') as f:
    np.save(f, np.array(normalize_statistics))

train_graph = MyDataset(x_list[0], label_list[0])
val_graph = MyDataset(x_list[1], label_list[1])
test_graph = MyDataset(x_list[2], label_list[2])

#loading the data for training, testing and validation
loader_tr = DataGenerator(train_graph, batch_size=args.batch_size, shuffle=True)
loader_va = DataGenerator(val_graph, batch_size=args.batch_size, shuffle=True)
loader_te = DataGenerator(test_graph, batch_size=args.batch_size, shuffle=True)

epochs = args.epoch
learning_rate = args.lr
dropout = args.dropout_rate

if __name__ == '__main__':
    try:
        before_start = datetime.now().timestamp()
        model = get_model(sample_graph)
        print(f'Model Architecture:\n{model.summary()}')
        h = model.fit(
            loader_tr,
            validation_data=loader_va,
            epochs=epochs, verbose = False,
        )
        #saving the model
        model_file = os.path.join(result_file, args.trial_name, 'trained_model')
        model.save(model_file, save_format = 'keras')
      
        after_end = datetime.now().timestamp()
        print(f'Completed in {(after_end - before_start) / 60} minutes')
      
    except KeyboardInterrupt:
        print('Model training stopped')
