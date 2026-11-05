# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 1 : entrainement + enregistrement du modele (CI/CD)

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_NAME = "workspace.sadiene2004_cicd.irisrfmodel"


def main(register_name=None, random_state=42):
    """Entraine le modele. Enregistre dans le registre seulement si register_name est donne."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=random_state
    )
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", accuracy)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "random_forest_model", signature=signature,
                                 serialization_format="cloudpickle")
        if register_name:
            mlflow.register_model(f"runs:/{run.info.run_id}/random_forest_model", register_name)
    return accuracy


# COMMAND ----------

if __name__ == "__main__":
    _spark = globals().get("spark")
    if _spark is not None:  # execute dans Databricks : registre Unity Catalog
        _spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.sadiene2004_cicd")
        mlflow.set_registry_uri("databricks-uc")
        print("accuracy =", main(register_name=MODEL_NAME))
    else:  # execute en local / CI : pas d'enregistrement
        print("accuracy =", main())
