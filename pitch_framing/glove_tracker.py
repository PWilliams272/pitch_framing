from ultralytics import YOLO
from roboflow import Roboflow
import os
from pathlib import Path
import shutil
import torch
from typing import Optional, Union, Any, List, Dict
import re


class GloveTracker:
    """
    Glove Tracker class
    """
    def __init__(
        self,
        model_location: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        dataset_location: Optional[Union[str, Path]] = None,
        dataset_name: Optional[str] = None,
        model_pt_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize the GloveTracker with a YOLOv8 model.

        Args:
            model_location: Path to the directory containing models. Defaults
                to ~/.pitch_framing/models.
            model_name: Name of the model to use or create.
            dataset_location: Path to the directory containing datasets.
                Defaults to ~/.pitch_framing/datasets.
            dataset_name: Name of the dataset to use.
            model_pt_path: Path to a specific model weights file (.pt). If
                provided, overrides model_location/model_name.
        """
        if model_location:
            self.model_location = Path(model_location)
        else:
            self.model_location = Path.home() / ".pitch_framing" / "models"
        self.model_name = model_name
        if dataset_location:
            self.dataset_location = Path(dataset_location)
        else:
            self.dataset_location = (
                Path.home() / ".pitch_framing" / "datasets"
            )
        os.makedirs(self.dataset_location, exist_ok=True)
        self.dataset_name = dataset_name
        self.model_pt_path = model_pt_path
        if self.model_pt_path is not None:
            self.model = YOLO(self.model_pt_path)
        else:
            self.model = None

    def _update_config(
        self,
        **kwargs
    ) -> Dict[str, Union[str, Path]]:
        """
        Helper function to handle updates to configuration parameters and
        instance member variables. If an argument is provided, it takes
        precedence over the existing member variable and will overwrite it.
        Also handles typing for path-like objects.
        Returns a dictionary of the current configuration after updates,
        which can be used
        """
        _PATH_KEYS = ["model_location", "dataset_location", "model_pt_path"]
        config = {}
        for key, value in kwargs.items():
            if value is None:
                config[key] = getattr(self, key)
                continue
            if key in _PATH_KEYS and not isinstance(value, Path):
                value = Path(value)
            setattr(self, key, value)
            config[key] = getattr(self, key)
        return config

    def check_available_datasets(
        self,
        dataset_location: Optional[Union[str, Path]] = None
    ) -> List[str]:
        """
        Helper function to check the datasets downloaded and available for
        training.

        Args:
            dataset_location: The directory to check for available datasets.
            Defaults to self.dataset_location.

        Returns:
            List[str]: List of dataset names found in the directory.
        """
        dataset_location = dataset_location or self.dataset_location
        if not os.path.exists(dataset_location):
            print(
                f"No datasets found in {dataset_location}. "
                f"Please download a dataset first."
            )
            return []
        print(f"Datasets found in {dataset_location}:")
        for dataset in os.listdir(dataset_location):
            print(f" - {dataset}")
        return os.listdir(dataset_location)

    def check_available_models(
        self,
        model_location: Optional[Union[str, Path]] = None
    ) -> List[str]:
        """
        Helper function to check the available trained models

        Args:
            model_location: The directory to check for available models.
                Defaults to self.model_location.

        Returns:
            List[str]: List of model names found in the directory.
        """
        model_location = model_location or self.model_location
        if not os.path.exists(model_location):
            print(
                f"No models found in {model_location}. "
                f"Please download a model first."
            )
            return []
        print(f"Models found in {model_location}:")
        for model in os.listdir(model_location):
            print(f" - {model}")
        return os.listdir(model_location)

    def download_roboflow_dataset(
        self,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        version: Optional[int] = None,
        dataset_location: Optional[Union[str, Path]] = None,
        dataset_format: str = 'yolov8'
    ) -> None:
        """
        Download annotated dataset from Roboflow for model training. Requires
        Roboflow API key stored as an environment variable.

        Args:
            workspace: The Roboflow workspace name (or set as environment
                variable "ROBOFLOW_WORKSPACE").
            project: The Roboflow project name (or set as environment variable
                "ROBOFLOW_PROJECT").
            version: The version number of the dataset (or set as environment
                variable "ROBOFLOW_PROJECT_VERSION").
            dataset_location: The directory to save the dataset. If None, will
                default to self.dataset_location.
        """
        self._update_config(dataset_location=dataset_location)
        os.makedirs(self.dataset_location, exist_ok=True)

        roboflow = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
        workspace = roboflow.workspace(
            workspace or os.environ['ROBOFLOW_WORKSPACE']
        )
        project = workspace.project(
            project or os.environ['ROBOFLOW_PROJECT']
        )
        version = project.version(
            version or os.environ['ROBOFLOW_PROJECT_VERSION']
        )
        # The "location" argument for the roboflow api isn't working, so
        # let it download to the current directory and then manually move it
        dataset = version.download(
            dataset_format,
            # location=self.dataset_location
        )
        self.dataset_name = Path(dataset.location).name
        destination = self.dataset_location / self.dataset_name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        shutil.move(dataset.location, destination)
        print(f"Dataset downloaded successfully to {destination}.")

    def train_model(
        self,
        model_location: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        base_model: str = 'yolov8n.pt',
        epochs: int = 50,
        lr0: float = 0.01,
        optimizer: str = 'Adam',
        dataset_location: Optional[Union[str, Path]] = None,
        dataset_name: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Train a YOLOv8 model on the specified dataset.

        Args:
            model_location: Directory to save the trained model. Defaults
                to self.model_location.
            model_name: Name for the trained model. If None, a new name is
                generated.
            base_model: Path or name of the base YOLOv8 model to fine-tune.
                Defaults to 'yolov8n.pt'.
            epochs: Number of training epochs. Defaults to 50.
            lr0: Initial learning rate. Defaults to 0.01.
            optimizer: Optimizer to use. Defaults to 'Adam'.
            dataset_location: Directory containing the dataset. Defaults to
                self.dataset_location.
            dataset_name: Name of the dataset. Defaults to self.dataset_name.
            **kwargs: Additional arguments passed to YOLO.train().
        """
        self._update_config(
            model_location=model_location,
            model_name=model_name
        )
        if self.model_name is None:
            available_models = self.check_available_models(
                self.model_location
            )
            run_nums = [
                int(m[4:]) for m in available_models if
                re.fullmatch(r"run_\d+", m)
            ]
            next_num = max(run_nums) + 1 if run_nums else 1
            self.model_name = f"run_{next_num}"
            print(f"Creating model: {self.model_name}")
        if dataset_location:
            self.dataset_location = Path(dataset_location)
        if dataset_name:
            self.dataset_name = dataset_name
        data_yaml = self.dataset_location / self.dataset_name / "data.yaml"
        self.model = YOLO(base_model)
        self.model_results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            lr0=lr0,
            optimizer=optimizer,
            project=self.model_location,
            name=self.model_name,
            **kwargs
        )
        self.model_name = self.model_results.save_dir.name

    def load_model(
        self,
        model_location: Union[str, Path],
        model_name: str,
    ) -> None:
        """
        Load a trained YOLOv8 model from the specified location.

        Args:
            model_location: Directory containing the trained model.
            model_name: Name of the model to load.
        """
        self._update_config(
            model_location=model_location,
            model_name=model_name
        )
        self.model = YOLO(
            self.model_location / self.model_name / "weights" / "best.pt"
        )

    def run_inference(
        self,
        video_path: Union[str, Path],
        model_pt_path: Optional[Union[str, Path]] = None,
        model_location: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
    ) -> Any:
        """
        Run inference on a video using the trained YOLOv8 model.

        Args:
            video_path: Path to the video file for inference.
            model_pt_path: Path to a specific model weights file (.pt). If
                provided, overrides model_location/model_name.
            model_location: Directory containing the model. Used if
                model_pt_path is not provided.
            model_name: Name of the model to use. Used if model_pt_path is
                not provided.

        Returns:
            Any: The results object from YOLO model inference.
        """
        self._update_config(
            model_location=model_location,
            model_name=model_name,
            model_pt_path=model_pt_path
        )
        if self.model_pt_path is None:
            self.model_pt_path = (
                self.model_location / self.model_name / "weights" / "best.pt"
            )
        self.model = YOLO(self.model_pt_path)
        self.model.eval()
        with torch.inference_mode():
            results = self.model(video_path)
        return results
