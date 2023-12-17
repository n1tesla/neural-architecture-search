import pandas as pd
import numpy as np
from data.data_loader import read_data
from joblib import dump
import os
#function for preparing training data and forming a RUL column with information about the remaining
# before breaking cycles

class DATA_PREPARATION:

    def __init__(self,config):
        self.config=config
        self.data_name = "FD001"
        self.id_features=config.id_features


    def scale_data(self, df_train=None, df_test=None,data_path=None):

        scaler_model = None

        if self.config.scaler_type == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler_model = RobustScaler()
        elif self.config.scaler_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler_model = MinMaxScaler()
        elif self.config.scaler_type == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler_model = StandardScaler()
        assert scaler_model is not None, f"Choose a valid scaler by using 'robust', 'minmax' or 'standard' keywords"

        # non_scale_cols = ['unit_number', 'time_in_cycles', 'RUL', 'label1', 'label2']
        # cols_normalize = self.df_train.columns.difference(non_scale_cols)
        df_train['cycle_norm'] = df_train['time_in_cycles']
        df_train.loc[:, self.config.features + ['cycle_norm']] = scaler_model.fit_transform(df_train.loc[:, self.config.features + ['cycle_norm']].values)
        df_test['cycle_norm'] = df_test['time_in_cycles']
        df_test.loc[:, self.config.features + ['cycle_norm']] = scaler_model.transform((df_test.loc[:, self.config.features + ['cycle_norm']].values))
        dump(scaler_model, open(os.path.join(data_path, "scaler.bin"), "wb"))

        return df_train,df_test

    def SlidingWindow(self, df: pd.core.frame.DataFrame) -> list:
        """
        @param df: Dataframe samples will be converted to Windowed array.
        @return: list of X_train and y_train
        """
        #Name of the column collected at different times and used to separate the cycles,
        # ex: 'unit_number' , 'split_index', 'different operational setting'
        data_split_name='unit_number'
        #target label name to be predicted
        target_label='RUL'
        WindowedX, WindowedY = [], []
        UniqueIds = df[data_split_name].unique()
        under_window_size=0
        for unique in UniqueIds:
            Split_Data = df[data_split_name].isin([unique])
            SplitDataX = df.loc[Split_Data, self.config.features].to_numpy()
            SplitDataY = df.loc[Split_Data, target_label].to_numpy()

            if SplitDataX.shape[0] >= self.config.window_size:
                WindowedX.append(extract_window(SplitDataX, self.config.window_size, self.config.stride_size))
                WindowedY.append(extract_labels(SplitDataY, self.config.window_size, self.config.stride_size))
            under_window_size+=1
        print(f"{under_window_size} samples are under window size")
        WindowedX = np.concatenate(WindowedX, axis=0)
        WindowedY = np.concatenate(WindowedY, axis=0)
        #shuffling windows in list
        idx = np.random.permutation(len(WindowedX))
        WindowedX, WindowedY = WindowedX[idx], WindowedY[idx]
        return [WindowedX, WindowedY]


    def create_dataset(self,data_path):

        df_train, df_test, df_rul = read_data(self.data_name)
        df_train = df_train.loc[:, self.id_features + self.config.features]
        df_test = df_test.loc[:, self.id_features + self.config.features]
        df_train = add_rul_column(df_train)
        df_test, df_truth = extract_max_rul(df_test, df_rul.copy())

        df_train,df_test=self.scale_data(df_train,df_test,data_path)
        dataset_dict= {'df_train': df_train, 'df_test': df_test}

        for k, v in dataset_dict.items():
            if k in ['df_train', 'df_test']:
                dataset_dict[k] = self.SlidingWindow(v)

        return dataset_dict['df_train'][0], dataset_dict['df_train'][1], dataset_dict['df_test'][0], dataset_dict['df_test'][1]



def extract_window(arr: np.ndarray, size: int, stride: int) -> np.ndarray:
    """
    @param arr: Unique Unit number
    @param size: Window size, also used in LSTM input shape (sample_size, feature_size,window_size)
    @param stride: Also called step size
    @return:
    """
    examples = []
    min_len = size - 1
    max_len = len(arr) - size
    for i in range(0, max_len + 1, stride):
        example = arr[i:size + i]
        examples.append(np.expand_dims(example, 0))

    return np.vstack(examples)
def extract_labels(arr: np.ndarray, size: int, stride: int) -> np.ndarray:
    """
    @param arr:
    @param size:
    @param stride:
    @return:
    """
    examples = []
    max_len = len(arr) - size
    for i in range(-1, max_len, stride):
        examples.append(arr[size + i])

    return np.array(examples)
def add_rul_column(data: pd.core.frame.DataFrame, factor: float = 0) -> pd.core.frame.DataFrame:
    df = data.copy()
    fd_RUL = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    fd_RUL = pd.DataFrame(fd_RUL)
    fd_RUL.columns = ['unit_number', 'max']
    df = df.merge(fd_RUL, on=['unit_number'], how='left')
    df['RUL'] = df['max'] - df['time_in_cycles']
    df.drop(columns=['max'], inplace=True)
    return df[df['time_in_cycles'] > factor]

def extract_max_rul(df_test: pd.core.frame.DataFrame, df_truth: pd.core.frame.DataFrame) -> tuple:
    rul = pd.DataFrame(df_test.groupby('unit_number')['time_in_cycles'].max()).reset_index()
    rul.columns = ['unit_number', 'max']
    df_truth.columns = ['more']
    df_truth['unit_number'] = df_truth.index + 1
    df_truth['max'] = rul['max'] + df_truth['more']  # adding true-rul vlaue + max cycle of test data set w.r.t MID
    df_truth.drop('more', axis=1, inplace=True)

    df_test = df_test.merge(df_truth, on=['unit_number'], how='left')
    df_test['RUL'] = df_test['max'] - df_test['time_in_cycles']
    df_test.drop('max', axis=1, inplace=True)
    return df_test, df_truth


# we will only make use of "label1" for binary classification,
# while trying to answer the question: is a specific engine going to fail within w1 cycles?
def convert_rul_to_label(df: pd.core.frame.DataFrame, w1: int = 30, w0: int = 15) -> pd.core.frame.DataFrame:
    df['label1'] = np.where(df['RUL'] <= w1, 1, 0)
    df['label2'] = df['label1']
    df.loc[df['RUL'] <= w0, 'label2'] = 2
    return df

    # TODO generate all labels under gen_labels function
    # label_gen = [gen_labels(train_df[train_df['unit_number']==id], sequence_length, ['RUL'])
    #              for id in train_df['unit_number'].unique()]
    #
    # label_array = np.concatenate(label_gen).astype(np.float32)


def gen_sequence(id_df, seq_length, seq_cols, slide_size):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(0, num_elements - seq_length, slide_size),
                           range(seq_length, num_elements, slide_size)):
        yield data_matrix[start:stop, :]
    # TODO: generate all sequence in gen_sequence


def gen_labels(id_df, seq_length, label, slide_size):
    labels = []
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]

    for i in range(seq_length, num_elements, slide_size):
        labels.append(data_matrix[i])
    return np.array(labels)

