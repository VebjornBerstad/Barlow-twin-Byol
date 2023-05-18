import os
from pathlib import Path


def main():
    # Set up directories.
    report_root = Path('.')
    gtzan_data_dir = (report_root / 'Data' / 'gtzan_val_mel_split').absolute()
    models_dir = (report_root / 'SavedModel').absolute()
    model_path = (models_dir / 'model.pth').absolute()
    linear_eval_model_path = (models_dir / 'linear_eval_model.pth').absolute()

    project_root = (report_root / 'Source' / 'Barlow-twins-that-listen/').absolute()

    # Change into the project dir, because we need some DVC dependencies.
    os.chdir(project_root)

    # Ensure venv is installed.
    if not (project_root / 'venv').exists():
        raise Exception("venv not installed. Please run `python -m venv venv` in the project root.")

    # Ensure that the venv is activated.
    if not os.environ.get('VIRTUAL_ENV'):
        raise Exception("venv not activated. Please run `source venv/bin/activate` (source venv/Scripts/activate on Windows) in the project root.")

    # Ensure that the swarm Python module is installed. Run pip freeze and check output.
    pip_freeze = os.popen("pip freeze").read()
    if "swarm" not in pip_freeze:
        raise Exception("swarm Python module not installed. Please run `pip install .` in the project root.")

    # Ensure data paths exists.
    if not gtzan_data_dir.exists():
        raise Exception(f"GTZAN data path does not exist: {gtzan_data_dir}. Please check that you have unzipped the data correctly.")

    if not model_path.exists():
        raise Exception(f"Model path does not exist: {model_path}. Please check that you have unzipped the model correctly.")

    if not linear_eval_model_path.exists():
        raise Exception(f"Linear eval model path does not exist: {linear_eval_model_path}. Please check that you have unzipped the model correctly.")

    # Run thte tests from the source directory.
    os.system(" ".join([
        "python -m swarm.test",
        "--gtzan_path_test", str(gtzan_data_dir),
        "--model_path", str(model_path),
        "--linear_eval_model_path", str(linear_eval_model_path),
    ]))


if __name__ == '__main__':
    main()
