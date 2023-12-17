from pathlib import Path
import pandas as pd

def read_data(data_name) -> tuple:
    """
    reads txt file under CMaps
    @return:
    """

    columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'TRA', 'FanInTemp', 'LPCompOT',
               'HPCompOT', 'LPTurbineOT', 'FanInP', 'BypassDuctP', 'HPCompOP', 'PhyFanSpeed', 'PhyCoreSpeed',
               'EngPreRatio', 'HPCOStaticPre', 'RFuelFlow', 'CorrectedFanSpeed', 'CorrectedCoreSpeed',
               'BypassRatio', 'BurnerFuelAirRatio', 'BleedEnthalpy', 'ReqFanSpeed', 'ReqFanConvSpeed',
               'HPTurbineCoolAirFlow', 'LPTurbineAirFlow']

    path_cwd = Path.cwd()
    train = pd.read_csv(path_cwd / f"CMaps/train_{data_name}.txt", sep=" ", header=None)
    test = pd.read_csv(path_cwd / f"CMaps/test_{data_name}.txt", sep=" ", header=None)
    RUL = pd.read_csv(path_cwd / f"CMaps/RUL_{data_name}.txt", sep=" ", header=None)

    # Nan columns
    train.drop(columns=[26, 27], inplace=True)
    test.drop(columns=[26, 27], inplace=True)
    RUL.drop(columns=[1], inplace=True)

    train.columns = columns
    test.columns = columns

    return train, test, RUL
