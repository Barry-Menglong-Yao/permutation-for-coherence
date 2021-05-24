from utils.constants import *

def gen_next_output_path(main_path):
    model_path = main_path / models_dir_name
    log_path = main_path / log_dir_name
    return model_path,log_path


def get_output_path_str(main_path):
    model_path = main_path / models_dir_name
    log_path = main_path / log_dir_name
    return str(model_path),str(log_path)