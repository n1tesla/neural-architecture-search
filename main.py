from preprocessing.prepare_data import DATA_PREPARATION
from configs.config import call_config,Parser
from util import utils
from train.train_2 import train

def start_search(config_all,observation):
    for network,input_cfg in config_all.items():
        config = Parser(input_cfg)
        observation_name= f"{config.network_name}_{observation}"
        start_time = utils.create_start_time()

        time_path, data_path, models_dir, plots_dir = utils.make_paths(observation_name, start_time)


        preperation=DATA_PREPARATION(config)
        X_train, y_train, X_test, y_test=preperation.create_dataset(data_path)
        train(config,X_train,y_train,X_test,y_test)


def main():
    config,observation=call_config()
    start_search(config,observation)


if __name__=="__main__":
    main()