import os
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.xgboost
import mlflow.lightgbm
from pathlib import Path
from config import config

class ModelRegistry:
    """Model registry for versioning and managing trained models."""

    def __init__(self, registry_uri=None):
        self.registry_uri = registry_uri or config.get('ml.tracking.tracking_uri')
        if self.registry_uri:
            mlflow.set_tracking_uri(self.registry_uri)

        self.experiment_name = config.get('ml.tracking.experiment_name', 'swmm_sensor_placement')

    def setup_experiment(self):
        """Set up MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            print(f"Warning: Could not set up MLflow experiment: {e}")

    def register_model(self, model, model_name, model_type="sklearn", **metadata):
        """Register a model in MLflow Model Registry."""
        try:
            self.setup_experiment()

            with mlflow.start_run() as run:
                # Log metadata
                for key, value in metadata.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                    else:
                        mlflow.log_param(key, str(value))

                # Log model based on type
                if model_type == "xgboost":
                    mlflow.xgboost.log_model(model, "model", registered_model_name=model_name)
                elif model_type == "lightgbm":
                    mlflow.lightgbm.log_model(model, "model", registered_model_name=model_name)
                elif model_type == "sklearn":
                    mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
                elif model_type == "pytorch":
                    mlflow.pytorch.log_model(model, "model", registered_model_name=model_name)
                else:
                    print(f"Warning: Unsupported model type {model_type} for registration")

                print(f"Model {model_name} registered successfully")

        except Exception as e:
            print(f"Warning: Could not register model {model_name}: {e}")

    def load_model(self, model_name, version="latest"):
        """Load a model from the registry."""
        try:
            if version == "latest":
                model_version = mlflow.get_latest_versions(model_name)[0]
            else:
                model_version = mlflow.get_model_version(model_name, version)

            model_uri = f"models:/{model_name}/{model_version.version}"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"Loaded model {model_name} version {model_version.version}")
            return model

        except Exception as e:
            print(f"Warning: Could not load model {model_name}: {e}")
            return None

    def list_models(self):
        """List all registered models."""
        try:
            client = mlflow.tracking.MlflowClient()
            return client.list_registered_models()
        except Exception as e:
            print(f"Warning: Could not list models: {e}")
            return []

    def transition_model_stage(self, model_name, version, stage):
        """Transition model to a different stage (None, Staging, Production, Archived)."""
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            print(f"Model {model_name} version {version} transitioned to {stage}")
        except Exception as e:
            print(f"Warning: Could not transition model stage: {e}")

# Global model registry instance
model_registry = ModelRegistry()</content>
<parameter name="filePath">c:\Users\kamlo\Desktop\Personal\projects\swmm_project\model_registry.py