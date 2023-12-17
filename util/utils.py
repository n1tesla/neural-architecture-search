from pathlib import Path
from typing import Tuple, Type, Callable
import os
from configs.config import Bcolors
def show_configurations(parameter_dict:dict):
    for key, value in parameter_dict.items():
        print(f"{Bcolors.WARNING}{key:<15}{Bcolors.ENDC}{value}")

def make_paths(observation_name: str, start_time: str) -> Tuple[Path, Path, Path, Path]:
    """
    Create directory structure for saving models, data, and plots.

    Parameters
    ----------
    observation_name : str
        The name of the observation.
    start_time : str
        The start time used to create a time-specific directory.

    Returns
    -------
    Tuple[Path, Path, Path, Path]
        A tuple containing the paths to time directory, data directory, models directory, and plots directory.
    """

    saved_models_path=Path.cwd() / "saved_models"
    saved_models_path.mkdir(exist_ok=True)

    observation_path= saved_models_path / observation_name

    if not os.path.exists(saved_models_path):
        saved_models_path.mkdir(exist_ok=True,parents=True)

    if not os.path.exists(observation_path):
        observation_path.mkdir(exist_ok=True)

    time_path=observation_path / start_time
    time_path.mkdir(exist_ok=True)

    data_path=time_path / "data"
    data_path.mkdir(exist_ok=True)

    models_dir=time_path/ "models"
    models_dir.mkdir(exist_ok=True)

    plots_dir=time_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    return time_path,data_path,models_dir,plots_dir


def create_start_time():
    import datetime
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    return start_time


def get_model_class(problem_type: str, module_name: str, model_name: str) -> Type:
    """
    Retrieve a models class based on the specified problem type, module, and models name.

    Parameters
    ----------
    problem_type : str
        The type of the problem (e.g., "classification", "regression", "forecasting").
    module_name : str
        The name of the module containing the models class.
    model_name : str
        The name of the models class.

    Returns
    -------
    Type
        The models class specified by the problem type, module, and models name.
    """
    import importlib

    try:
        imported_module = importlib.import_module(f"models.{problem_type}.{module_name}")
        model = getattr(imported_module, model_name)
        return model
    except ImportError as e:
        raise ImportError(f"Error importing module: {e}")
    except AttributeError as e:
        raise AttributeError(f"Error retrieving models class: {e}")

