import os


def assert_is_directory(filepath: str, err_msg: str):
    assert_path_exists(filepath, err_msg)
    if not os.path.isdir(filepath):
        print(f"'{filepath}' must be a directory. " + err_msg)
        raise ValueError()


def assert_path_exists(filepath: str, err_msg: str):
    if not os.path.exists(filepath):
        print(f"No file or directory found at '{filepath}'. " + err_msg)
        raise ValueError()
