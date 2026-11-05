import mlflow
from notebooks.train_model import main


def _setup(tmp_path):
    db = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    return db


def test_model_training(tmp_path):
    db = _setup(tmp_path)
    accuracy = main()
    assert db.exists(), "MLflow run not found"
    assert len(mlflow.search_runs(search_all_experiments=True)) == 1
    assert accuracy >= 0.85


def test_accuracy_between_0_and_1(tmp_path):
    _setup(tmp_path)
    assert 0.0 <= main() <= 1.0
