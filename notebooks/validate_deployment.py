# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 2 : validation du deploiement

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
MODEL_NAME = "workspace.sadiene2004_cicd.irisrfmodel"

versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
for v in sorted(versions, key=lambda v: int(v.version)):
    print("version", v.version, "| cree le", v.creation_timestamp, "| run", v.run_id)
assert len(versions) >= 1, "Aucune version enregistree"

# COMMAND ----------

import pandas as pd
from sklearn.datasets import load_iris

latest = max(int(v.version) for v in versions)
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{latest}")
X = load_iris().data[[0, 60, 120]]
print("predictions (attendu 0, 1, 2) :", model.predict(pd.DataFrame(X)).tolist())
