from ultralytics import YOLO
from roboflow import Roboflow
import os
from pathlib import Path
import shutil
import torch
from typing import Optional, Union, Any, List, Dict
import re
import json


class GloveTracker:
    """Orchestrates glove tracking modeling.

    Class handles downloading and managing datasets from Roboflow,
    training models, making predictions.

    Attributes:
        model_location (Path): Path to the directory containing models.
        model_name (str): Name of the model to use or create.
        dataset_location (Path): Path to the directory containing datasets.
        dataset_name (str): Name of the dataset to use.
        model_pt_path (Path): Path to a specific model weights file (.pt).
        model (YOLO): The YOLO model instance.
    """
    def __init__(
        self,
        model_location: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        model_pt_path: Optional[Union[str, Path]] = None,
        dataset_location: Optional[Union[str, Path]] = None,
        dataset_name: Optional[str] = None,
        inference_location: Optional[Union[str, Path]] = None
    ) -> None:
        """Initialize GloveTracker

        Args:
            model_location: Path to the directory containing models. Defaults
                to ~/.pitch_framing/models.
            model_name: Name of the model to use or create.
            model_pt_path: Path to a specific model weights file (.pt). If
                provided, will automatically load in the model.
            dataset_location: Path to the directory containing datasets.
                Defaults to ~/.pitch_framing/datasets.
            dataset_name: Name of the dataset to use.
            inference_location: Path to the directory containing inference
                results.
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

        if inference_location:
            self.inference_location = Path(inference_location)
        else:
            self.inference_location = (
                Path.home() / ".pitch_framing" / "inference"
            )
        os.makedirs(self.inference_location, exist_ok=True)

    def _update_config(
        self,
        **kwargs
    ) -> Dict[str, Union[str, Path]]:
        """Update configuration parameters.

        Helper function to handle updates to configuration parameters and
        instance member variables. If an argument is provided, it takes
        precedence over the existing member variable and will overwrite it.
        Also handles typing for path-like objects.
        Returns a dictionary of the current configuration after updates.

        Args:
            **kwargs: Configuration parameters to update.

        Returns:
            Dict[str, Union[str, Path]]: The updated configuration.
        """
        _PATH_KEYS = [
            "model_location",
            "dataset_location",
            "model_pt_path",
            "inference_location"
        ]
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
        """Check available datasets.

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
        """Check available models.

        Helper function to check the available trained models.

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
        """Download annotated Roboflow dataset.

        Downloads an annotated dataset from Roboflow for model training.
        Requires Roboflow API key stored as an environment variable, other
        necessary parameters (workspace, project, version) can be set as
        environment variables or passed as arguments.

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
        """Train a YOLO model.

        Fine tunes a YOLO model using the specified dataset and training
        parameters. By default, will use the pretrained yolov8n.pt model
        as the initial weights. Additional training parameters can be
        specified as keyword arguments.

        Args:
            model_location: Directory to save the trained model. Defaults
                to self.model_location.
            model_name: Name for the trained model. If None, a new name is
                generated.
            base_model: Path or name of the base YOLO model to fine-tune.
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
        model_location: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        model_pt_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Load a trained YOLO model.

        Args:
            model_location: Directory containing the trained model.
            model_name: Name of the model to load.
            model_pt_path: Path to the model weights file (.pt).
        """
        self._update_config(
            model_location=model_location,
            model_name=model_name,
            model_pt_path=model_pt_path
        )
        # Require either model_name or model_pt_path to load a model
        if self.model_name is None and self.model_pt_path is None:
            raise AssertionError(
                "model_name or model_pt_path must be specified "
                "to load a model"
            )
        # If model_pt_path is not specified, use the default path
        if self.model_pt_path is None:
            self.model_pt_path = (
                self.model_location / self.model_name / "weights" / "best.pt"
            )
        self.model = YOLO(self.model_pt_path)

    def yolo_results_to_json(
        self,
        results
    ):
        """Convert YOLO inference results to JSON format.

        Args:
            results: The inference results to save.
        """
        json_results = []
        for frame, _result in enumerate(results):
            box_list = []
            for box in _result.boxes:
                xyxy = box.xyxy.cpu().numpy().flatten()
                box_list.append({
                    "class": int(box.cls[0]),
                    "x_center": float((xyxy[0] + xyxy[2]) / 2),
                    "y_center": float((xyxy[1] + xyxy[3]) / 2),
                    "width": float(xyxy[2] - xyxy[0]),
                    "height": float(xyxy[3] - xyxy[1]),
                    "confidence": float(box.conf[0])
                })
            json_results.append({
                "frame": frame,
                "boxes": box_list
            })
        return json_results

    def save_results_json(
        self,
        results: Any,
        filename: Union[str, Path]
    ) -> None:
        """Save YOLO inference results to a JSON file.

        Args:
            results: The inference results to save.
            filename: The path to the file where results will be saved.
        """
        json_results = self.yolo_results_to_json(results)
        with open(filename, "w") as f:
            json.dump(json_results, f, indent=2)

    def run_inference(
        self,
        video_path: Union[str, Path, List[Union[str, Path]]],
        model_pt_path: Optional[Union[str, Path]] = None,
        model_location: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        save_results: bool = False,
        inference_location: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Run inference on a video.

        Args:
            video_path: Path or list of paths to the video file(s) for
                inference.
            model_pt_path: Path to a specific model weights file (.pt). If
                provided, overrides model_location/model_name.
            model_location: Directory containing the model. Used if
                model_pt_path is not provided.
            model_name: Name of the model to use. Used if model_pt_path is
                not provided.
            save_results: Whether to save the inference results. If True,
                results will be saved to the inference_location.
            inference_location: Path to the directory where inference results
                will be saved. If not provided, defaults to
                ~/.pitch_framing/inference.

        Returns:
            Any: The results object from YOLO model inference.
        """
        self._update_config(
            model_location=model_location,
            model_name=model_name,
            model_pt_path=model_pt_path,
            inference_location=inference_location
        )
        if self.model is None:
            self.load_model()

        # If a list, recursively call method
        if isinstance(video_path, list):
            results = [
                self.run_inference(
                    video_path=video,
                    save_results=save_results
                )
                for video in video_path
            ]
            return results

        # Set model to evaluation mode and disable torch gradient calculations
        self.model.eval()
        with torch.inference_mode():
            results = self.model(video_path, stream=True)
            if save_results:
                filename = (
                    self.inference_location
                    / f"{Path(video_path).stem}_results.json"
                )
                self.save_results_json(results, filename)
        return results
