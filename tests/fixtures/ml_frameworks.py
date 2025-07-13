"""
Fixtures for testing ML framework dependency patterns.

This module creates realistic ML projects using popular frameworks like
TensorFlow, PyTorch, Hugging Face Transformers, and scientific computing
libraries to test complex dependency detection scenarios.
"""

from pathlib import Path


class MLFrameworksFixture:
    """Creates ML framework project structures for testing."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)

    def create_tensorflow_project(self) -> Path:
        """Create a TensorFlow/Keras project with complex dependencies."""
        tf_project = self.base_path / "tensorflow_project"
        tf_project.mkdir(parents=True, exist_ok=True)

        # Main training script
        (tf_project / "train.py").write_text('''
"""TensorFlow training script with complex import patterns."""

# Core TensorFlow imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# TensorFlow extended ecosystem
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import tensorflow_hub as hub
import tensorflow_model_optimization as tfmot

# TensorFlow serving and deployment
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorboard
from tensorboard.plugins import projector

# Scientific computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Computer vision specific
import cv2
import PIL
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2

# MLflow integration
import mlflow
import mlflow.tensorflow
import mlflow.keras

# Utilities
import json
import yaml
import logging
import argparse
from pathlib import Path
import warnings


class TensorFlowModelTrainer:
    """Advanced TensorFlow model trainer with multiple frameworks."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.model = None
        self.data_generator = None

        # TensorFlow configuration
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logging.warning(f"GPU configuration failed: {e}")

    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration."""
        with open(config_path) as f:
            if config_path.endswith('.yaml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)

    def create_model(self) -> tf.keras.Model:
        """Create model with transfer learning."""

        # Base model selection
        if self.config['base_model'] == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
        elif self.config['base_model'] == 'mobilenet':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
        else:
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )

        # Freeze base model
        base_model.trainable = False

        # Add custom layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.config['num_classes'], activation='softmax')
        ])

        # Compile with advanced optimizers
        if self.config['optimizer'] == 'adamw':
            optimizer = tfa.optimizers.AdamW(
                learning_rate=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        else:
            optimizer = optimizers.Adam(
                learning_rate=self.config['learning_rate']
            )

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )

        return model

    def create_data_pipeline(self) -> tf.data.Dataset:
        """Create TensorFlow data pipeline with augmentations."""

        # Load dataset using TensorFlow Datasets
        if self.config['dataset'] == 'cifar10':
            (ds_train, ds_test), ds_info = tfds.load(
                'cifar10',
                split=['train', 'test'],
                with_info=True,
                as_supervised=True
            )
        else:
            # Custom dataset loading
            ds_train = tf.data.Dataset.from_tensor_slices(
                (self.config['train_images'], self.config['train_labels'])
            )

        # Data augmentation using Albumentations
        def augment_image(image, label):
            # Convert to numpy for albumentations
            image_np = image.numpy()

            transform = A.Compose([
                A.RandomRotate90(),
                A.Flip(),
                A.Transpose(),
                A.GaussNoise(p=0.2),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.1),
                    A.PiecewiseAffine(p=0.3),
                ], p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

            augmented = transform(image=image_np)
            return tf.convert_to_tensor(augmented['image']), label

        # Apply augmentations
        ds_train = ds_train.map(
            lambda x, y: tf.py_function(augment_image, [x, y], [tf.float32, tf.int64]),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Batch and prefetch
        ds_train = ds_train.batch(self.config['batch_size']).prefetch(tf.data.AUTOTUNE)

        return ds_train

    def train(self) -> tf.keras.Model:
        """Train the model with advanced callbacks."""

        # Create model and data
        self.model = self.create_model()
        train_dataset = self.create_data_pipeline()

        # Advanced callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]

        # Mixed precision training
        if self.config['use_mixed_precision']:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config)

            # Train model
            history = self.model.fit(
                train_dataset,
                epochs=self.config['epochs'],
                callbacks=callbacks_list,
                validation_split=0.2
            )

            # Log metrics
            final_accuracy = max(history.history['val_accuracy'])
            mlflow.log_metric("final_accuracy", final_accuracy)

            # Log model
            mlflow.tensorflow.log_model(self.model, "tensorflow_model")

            # Model optimization for deployment
            if self.config['optimize_for_inference']:
                # TensorFlow Lite conversion
                converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()

                with open('model.tflite', 'wb') as f:
                    f.write(tflite_model)

                mlflow.log_artifact('model.tflite')

                # TensorFlow.js conversion
                import tensorflowjs as tfjs
                tfjs.converters.save_keras_model(self.model, 'tfjs_model')
                mlflow.log_artifacts('tfjs_model')

        return self.model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()

    trainer = TensorFlowModelTrainer(args.config)
    model = trainer.train()
''')

        # TensorFlow Serving script
        (tf_project / "serve.py").write_text('''
"""TensorFlow Serving deployment script."""

import tensorflow as tf
import tensorflow_serving.apis.predict_pb2 as predict_pb2
import tensorflow_serving.apis.prediction_service_pb2_grpc as prediction_service_pb2_grpc
import grpc
import numpy as np
from flask import Flask, request, jsonify
import docker
import kubernetes
from kubernetes import client, config as k8s_config

class TensorFlowServingClient:
    """Client for TensorFlow Serving."""

    def __init__(self, server_url: str):
        self.channel = grpc.insecure_channel(server_url)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)

    def predict(self, data: np.ndarray, model_name: str) -> np.ndarray:
        """Make prediction using TensorFlow Serving."""

        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.model_spec.signature_name = 'serving_default'
        request.inputs['input'].CopyFrom(tf.make_tensor_proto(data))

        response = self.stub.Predict(request, 10.0)
        return tf.make_ndarray(response.outputs['output'])


class KubernetesDeployer:
    """Deploy TensorFlow models to Kubernetes."""

    def __init__(self):
        k8s_config.load_incluster_config()
        self.v1 = client.AppsV1Api()

    def deploy_model(self, model_name: str, model_version: str):
        """Deploy TensorFlow Serving to Kubernetes."""

        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(name=f"tf-serving-{model_name}"),
            spec=client.V1DeploymentSpec(
                replicas=3,
                selector=client.V1LabelSelector(
                    match_labels={"app": f"tf-serving-{model_name}"}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": f"tf-serving-{model_name}"}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="tensorflow-serving",
                                image="tensorflow/serving:latest-gpu",
                                ports=[client.V1ContainerPort(container_port=8501)],
                                env=[
                                    client.V1EnvVar(
                                        name="MODEL_NAME",
                                        value=model_name
                                    )
                                ]
                            )
                        ]
                    )
                )
            )
        )

        self.v1.create_namespaced_deployment(
            namespace="default",
            body=deployment
        )


app = Flask(__name__)
serving_client = TensorFlowServingClient("localhost:8500")

@app.route("/predict", methods=["POST"])
def predict():
    """Flask endpoint for predictions."""
    data = request.json
    input_data = np.array(data["inputs"])

    predictions = serving_client.predict(input_data, "my_model")

    return jsonify({
        "predictions": predictions.tolist()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
''')

        # Configuration file
        (tf_project / "config.yaml").write_text("""
base_model: "resnet50"
num_classes: 10
batch_size: 32
epochs: 50
learning_rate: 0.001
weight_decay: 0.0001
optimizer: "adamw"
dataset: "cifar10"
use_mixed_precision: true
optimize_for_inference: true
""")

        return tf_project

    def create_pytorch_project(self) -> Path:
        """Create a PyTorch project with complex dependencies."""
        pytorch_project = self.base_path / "pytorch_project"
        pytorch_project.mkdir(parents=True, exist_ok=True)

        # Main training script
        (pytorch_project / "train.py").write_text('''
"""PyTorch training script with advanced features."""

# Core PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

# PyTorch ecosystem
import torchvision
import torchvision.transforms as transforms
from torchvision import models, datasets
import torchmetrics
from torchmetrics import Accuracy, F1Score, AUROC

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin

# Transformers and NLP
import transformers
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertModel, GPT2Model, T5Model,
    Trainer as HFTrainer, TrainingArguments,
    DataCollatorWithPadding
)
import datasets
from datasets import load_dataset, Dataset as HFDataset

# Computer Vision
import timm
from efficientnet_pytorch import EfficientNet
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Optimization and utilities
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import wandb
from apex import amp
import fairscale
from fairscale.nn import auto_wrap, default_auto_wrap_policy
from fairscale.optim.oss import OSS

# Scientific computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# MLflow integration
import mlflow
import mlflow.pytorch

# Utilities
import json
import yaml
import logging
import argparse
from pathlib import Path
import warnings
import os
from typing import Dict, List, Optional, Tuple, Any


class AdvancedPyTorchModel(LightningModule):
    """Advanced PyTorch Lightning model with multiple architectures."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        # Model architecture selection
        if config['architecture'] == 'resnet':
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, config['num_classes'])
        elif config['architecture'] == 'efficientnet':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
            self.backbone._fc = nn.Linear(self.backbone._fc.in_features, config['num_classes'])
        elif config['architecture'] == 'timm':
            self.backbone = timm.create_model(
                config['timm_model'],
                pretrained=True,
                num_classes=config['num_classes']
            )
        elif config['architecture'] == 'transformer':
            self.backbone = AutoModel.from_pretrained(config['transformer_model'])
            self.classifier = nn.Linear(self.backbone.config.hidden_size, config['num_classes'])

        # Loss function
        if config['loss'] == 'focal':
            from torchvision.ops import focal_loss
            self.criterion = lambda x, y: focal_loss.sigmoid_focal_loss(
                x, y, alpha=config['focal_alpha'], gamma=config['focal_gamma']
            )
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.0))

        # Metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=config['num_classes'])
        self.val_accuracy = Accuracy(task='multiclass', num_classes=config['num_classes'])
        self.val_f1 = F1Score(task='multiclass', num_classes=config['num_classes'])

    def forward(self, x):
        """Forward pass."""
        if self.config['architecture'] == 'transformer':
            outputs = self.backbone(x)
            return self.classifier(outputs.pooler_output)
        else:
            return self.backbone(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log metrics
        self.train_accuracy(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log metrics
        self.val_accuracy(logits, y)
        self.val_f1(logits, y)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', self.val_accuracy, on_epoch=True)
        self.log('val_f1', self.val_f1, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""

        # Optimizer selection
        if self.config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'lamb':
            from pytorch_lamb import Lamb
            optimizer = Lamb(
                self.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )

        # Scheduler selection
        if self.config['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['epochs']
            )
        elif self.config['scheduler'] == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1
            )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_acc' if self.config['scheduler'] == 'reduce_on_plateau' else None
            }
        }


class PyTorchDataModule(LightningDataModule):
    """PyTorch Lightning data module."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.transform_train = A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        self.transform_val = A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if self.config['dataset'] == 'cifar10':
            self.train_dataset = datasets.CIFAR10(
                root='./data',
                train=True,
                download=True,
                transform=self.transform_train
            )
            self.val_dataset = datasets.CIFAR10(
                root='./data',
                train=False,
                download=True,
                transform=self.transform_val
            )
        elif self.config['dataset'] == 'imagenet':
            self.train_dataset = datasets.ImageNet(
                root='./data/imagenet',
                split='train',
                transform=self.transform_train
            )
            self.val_dataset = datasets.ImageNet(
                root='./data/imagenet',
                split='val',
                transform=self.transform_val
            )

    def train_dataloader(self):
        """Training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

    def val_dataloader(self):
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )


class HyperparameterOptimizer:
    """Hyperparameter optimization with Optuna."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def objective(self, trial):
        """Optuna objective function."""

        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

        # Update config
        config = self.config.copy()
        config.update({
            'learning_rate': lr,
            'batch_size': batch_size,
            'weight_decay': weight_decay
        })

        # Create model and data module
        model = AdvancedPyTorchModel(config)
        data_module = PyTorchDataModule(config)

        # Create trainer with pruning callback
        trainer = Trainer(
            max_epochs=10,  # Reduced for optimization
            callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_acc')],
            logger=False,
            enable_checkpointing=False,
            accelerator='auto'
        )

        # Train model
        trainer.fit(model, data_module)

        # Return validation accuracy
        return trainer.callback_metrics['val_acc'].item()

    def optimize(self, n_trials: int = 100):
        """Run hyperparameter optimization."""
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)

        return study.best_params


def train_distributed_model(rank, world_size, config):
    """Distributed training function."""

    # Initialize distributed training
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Create model and wrap with DDP
    model = AdvancedPyTorchModel(config)
    model = DDP(model.cuda(rank), device_ids=[rank])

    # Create data module with distributed sampler
    data_module = PyTorchDataModule(config)
    data_module.setup()

    train_sampler = DistributedSampler(
        data_module.train_dataset,
        num_replicas=world_size,
        rank=rank
    )

    train_loader = DataLoader(
        data_module.train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # Training loop
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['epochs']):
        train_sampler.set_epoch(epoch)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(rank), target.cuda(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0 and rank == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

    dist.destroy_process_group()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.optimize:
        # Run hyperparameter optimization
        optimizer = HyperparameterOptimizer(config)
        best_params = optimizer.optimize(n_trials=50)
        print(f"Best hyperparameters: {best_params}")

        # Update config with best parameters
        config.update(best_params)

    if args.distributed:
        # Distributed training
        world_size = torch.cuda.device_count()
        mp.spawn(train_distributed_model, args=(world_size, config), nprocs=world_size, join=True)
    else:
        # Single GPU/CPU training
        model = AdvancedPyTorchModel(config)
        data_module = PyTorchDataModule(config)

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                monitor='val_acc',
                mode='max',
                save_top_k=1,
                filename='best-{epoch:02d}-{val_acc:.2f}'
            ),
            EarlyStopping(
                monitor='val_acc',
                mode='max',
                patience=10
            ),
            LearningRateMonitor(logging_interval='step')
        ]

        # Logger
        logger = TensorBoardLogger('lightning_logs', name='pytorch_model')

        # Trainer
        trainer = Trainer(
            max_epochs=config['epochs'],
            callbacks=callbacks,
            logger=logger,
            accelerator='auto',
            precision=16 if config.get('use_amp', False) else 32,
            strategy='ddp' if torch.cuda.device_count() > 1 else None
        )

        # MLflow tracking
        with mlflow.start_run():
            mlflow.log_params(config)

            # Train model
            trainer.fit(model, data_module)

            # Log final metrics
            mlflow.log_metric("final_val_acc", trainer.callback_metrics['val_acc'].item())

            # Log model
            mlflow.pytorch.log_model(model, "pytorch_model")


if __name__ == "__main__":
    main()
''')

        # Configuration file
        (pytorch_project / "config.yaml").write_text("""
architecture: "resnet"
num_classes: 10
batch_size: 32
epochs: 50
learning_rate: 0.001
weight_decay: 0.0001
optimizer: "adamw"
scheduler: "cosine"
dataset: "cifar10"
num_workers: 4
use_amp: true
loss: "cross_entropy"
label_smoothing: 0.1
""")

        # Requirements file
        (pytorch_project / "requirements.txt").write_text("""
torch>=1.12.0
torchvision>=0.13.0
pytorch-lightning>=1.7.0
transformers>=4.20.0
datasets>=2.0.0
timm>=0.6.0
efficientnet-pytorch>=0.7.0
segmentation-models-pytorch>=0.3.0
albumentations>=1.2.0
torchmetrics>=0.9.0
optuna>=3.0.0
wandb>=0.13.0
apex
fairscale>=0.4.0
pytorch-lamb>=1.2.0
mlflow>=1.28.0
""")

        return pytorch_project

    def create_huggingface_project(self) -> Path:
        """Create a Hugging Face Transformers project."""
        hf_project = self.base_path / "huggingface_project"
        hf_project.mkdir(parents=True, exist_ok=True)

        # Main NLP training script
        (hf_project / "train_nlp.py").write_text('''
"""Hugging Face Transformers training script with advanced features."""

# Core Transformers imports
import transformers
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    AutoModelForSequenceClassification, AutoModelForQuestionAnswering,
    AutoModelForTokenClassification, AutoModelForCausalLM,
    BertTokenizer, BertModel, BertForSequenceClassification,
    RobertaTokenizer, RobertaModel, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification,
    GPT2Tokenizer, GPT2Model, GPT2LMHeadModel,
    T5Tokenizer, T5Model, T5ForConditionalGeneration,
    BartTokenizer, BartModel, BartForConditionalGeneration,
    Trainer, TrainingArguments,
    DataCollatorWithPadding, DataCollatorForLanguageModeling,
    pipeline, set_seed
)

# Datasets and evaluation
import datasets
from datasets import load_dataset, Dataset, DatasetDict, load_metric
import evaluate
from evaluate import load as load_metric

# Model optimization and efficiency
from transformers import (
    AdamW, get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW as TorchAdamW

# Advanced training techniques
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
import deepspeed
from deepspeed import zero

# Model compression and optimization
import optimum
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer
from optimum.intel import INCQuantizer
from transformers.optimization import Adafactor

# Specialized libraries
import sentencepiece
import tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# Scientific computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# MLflow integration
import mlflow
import mlflow.transformers

# Utilities
import json
import yaml
import logging
import argparse
from pathlib import Path
import warnings
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import wandb


@dataclass
class ModelConfig:
    """Configuration for transformer models."""
    model_name: str
    task: str  # classification, qa, ner, generation
    num_labels: Optional[int] = None
    max_length: int = 512
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    use_deepspeed: bool = False
    use_gradient_checkpointing: bool = False
    fp16: bool = True
    gradient_accumulation_steps: int = 1


class AdvancedTransformerTrainer:
    """Advanced trainer for transformer models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.dataset = None

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize accelerator for distributed training
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs]
        )

    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer based on task."""

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Select model class based on task
        if self.config.task == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels
            )
        elif self.config.task == "qa":
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                self.config.model_name
            )
        elif self.config.task == "ner":
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels
            )
        elif self.config.task == "generation":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name
            )
        else:
            raise ValueError(f"Unsupported task: {self.config.task}")

        # Apply LoRA if specified
        if self.config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS if self.config.task == "classification" else TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=0.1
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Enable gradient checkpointing for memory efficiency
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def load_and_preprocess_data(self, dataset_name: str, dataset_config: Optional[str] = None):
        """Load and preprocess dataset."""

        # Load dataset
        if dataset_name in ["imdb", "sst2", "cola", "mnli", "qnli", "qqp", "rte", "wnli", "mrpc", "stsb"]:
            # GLUE benchmarks
            self.dataset = load_dataset("glue", dataset_name)
        elif dataset_name == "squad":
            self.dataset = load_dataset("squad")
        elif dataset_name == "conll2003":
            self.dataset = load_dataset("conll2003")
        else:
            self.dataset = load_dataset(dataset_name, dataset_config)

        # Preprocessing based on task
        if self.config.task == "classification":
            self.dataset = self.dataset.map(
                self._preprocess_classification,
                batched=True,
                remove_columns=self.dataset["train"].column_names
            )
        elif self.config.task == "qa":
            self.dataset = self.dataset.map(
                self._preprocess_qa,
                batched=True
            )
        elif self.config.task == "ner":
            self.dataset = self.dataset.map(
                self._preprocess_ner,
                batched=True
            )

    def _preprocess_classification(self, examples):
        """Preprocess data for classification tasks."""
        if "sentence" in examples:
            texts = examples["sentence"]
        elif "text" in examples:
            texts = examples["text"]
        else:
            texts = [f"{ex1} {self.tokenizer.sep_token} {ex2}"
                    for ex1, ex2 in zip(examples["sentence1"], examples["sentence2"])]

        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )

    def _preprocess_qa(self, examples):
        """Preprocess data for question answering."""
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]

        inputs = self.tokenizer(
            questions,
            contexts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        # Process answers for training
        if "answers" in examples:
            start_positions = []
            end_positions = []

            for i, answer in enumerate(examples["answers"]):
                if len(answer["answer_start"]) == 0:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    start_char = answer["answer_start"][0]
                    end_char = start_char + len(answer["text"][0])

                    # Find token positions
                    token_start = None
                    token_end = None

                    for idx, (start, end) in enumerate(inputs["offset_mapping"][i]):
                        if start <= start_char < end:
                            token_start = idx
                        if start < end_char <= end:
                            token_end = idx
                            break

                    start_positions.append(token_start if token_start is not None else 0)
                    end_positions.append(token_end if token_end is not None else 0)

            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions

        return inputs

    def _preprocess_ner(self, examples):
        """Preprocess data for NER tasks."""
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding=True,
            max_length=self.config.max_length
        )

        # Align labels with tokens
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []

            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def train(self):
        """Train the model."""

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_dir="./logs",
            logging_steps=100,
            evaluation_strategy="steps" if "validation" in self.dataset else "no",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy" if self.config.task == "classification" else "eval_loss",
            greater_is_better=True if self.config.task == "classification" else False,
            fp16=self.config.fp16,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="wandb",
            run_name=f"{self.config.model_name}_{self.config.task}",
            deepspeed="ds_config.json" if self.config.use_deepspeed else None
        )

        # Data collator
        if self.config.task == "generation":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        else:
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred

            if self.config.task == "classification":
                predictions = np.argmax(predictions, axis=1)
                accuracy = accuracy_score(labels, predictions)
                f1 = f1_score(labels, predictions, average="weighted")
                return {"accuracy": accuracy, "f1": f1}
            else:
                return {}

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset.get("validation"),
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        # Start MLflow run
        with mlflow.start_run():
            # Log configuration
            mlflow.log_params(self.config.__dict__)

            # Train model
            train_result = trainer.train()

            # Log training metrics
            mlflow.log_metrics({
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.metrics["train_runtime"],
                "train_samples_per_second": train_result.metrics["train_samples_per_second"]
            })

            # Evaluate model
            if "test" in self.dataset:
                eval_result = trainer.evaluate(self.dataset["test"])
                mlflow.log_metrics({f"test_{k}": v for k, v in eval_result.items()})

            # Save model
            trainer.save_model("./final_model")
            mlflow.transformers.log_model(
                transformers_model=self.model,
                artifact_path="model",
                tokenizer=self.tokenizer
            )

        return trainer

    def optimize_model(self, model_path: str):
        """Optimize model for inference."""

        # ONNX optimization
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            model_path,
            from_transformers=True
        )

        optimizer = ORTOptimizer.from_pretrained(onnx_model)
        optimizer.optimize(save_dir="./optimized_model")

        # Quantization
        quantizer = INCQuantizer.from_pretrained(onnx_model)
        quantizer.quantize(
            save_dir="./quantized_model",
            calibration_dataset=self.dataset["train"].select(range(100))
        )

        return onnx_model


def create_deepspeed_config():
    """Create DeepSpeed configuration."""
    config = {
        "fp16": {
            "enabled": True,
            "auto_cast": True
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 2e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 2e-5,
                "warmup_num_steps": 500
            }
        },
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 4,
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False
    }

    with open("ds_config.json", "w") as f:
        json.dump(config, f, indent=2)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--model_name", required=True, help="Model name")
    parser.add_argument("--task", required=True, help="Task type")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    config_dict.update({
        "model_name": args.model_name,
        "task": args.task
    })

    config = ModelConfig(**config_dict)

    # Create DeepSpeed config if needed
    if config.use_deepspeed:
        create_deepspeed_config()

    # Initialize wandb
    wandb.init(
        project="transformer-training",
        config=config.__dict__,
        name=f"{config.model_name}_{config.task}"
    )

    # Create trainer
    trainer = AdvancedTransformerTrainer(config)

    # Setup model and data
    trainer.setup_model_and_tokenizer()
    trainer.load_and_preprocess_data(args.dataset)

    # Train model
    trained_model = trainer.train()

    # Optimize model
    trainer.optimize_model("./final_model")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
''')

        # Configuration file
        (hf_project / "config.yaml").write_text("""
num_labels: 2
max_length: 512
learning_rate: 2e-5
batch_size: 16
epochs: 3
warmup_steps: 500
weight_decay: 0.01
use_lora: false
lora_r: 8
lora_alpha: 32
use_deepspeed: false
use_gradient_checkpointing: true
fp16: true
gradient_accumulation_steps: 1
""")

        # Requirements file
        (hf_project / "requirements.txt").write_text("""
transformers>=4.21.0
datasets>=2.4.0
evaluate>=0.2.0
peft>=0.4.0
accelerate>=0.20.0
deepspeed>=0.9.0
optimum[onnxruntime]>=1.9.0
sentencepiece>=0.1.97
tokenizers>=0.12.0
wandb>=0.13.0
mlflow>=2.0.0
torch>=1.12.0
""")

        return hf_project

    def create_scientific_computing_project(self) -> Path:
        """Create a scientific computing project with complex dependencies."""
        sci_project = self.base_path / "scientific_computing_project"
        sci_project.mkdir(parents=True, exist_ok=True)

        # Main analysis script
        (sci_project / "analysis.py").write_text('''
"""Scientific computing analysis with complex dependency patterns."""

# Core scientific computing stack
import numpy as np
import pandas as pd
import scipy
from scipy import stats, optimize, interpolate, integrate, linalg, sparse

# Advanced numerical computing
import numba
import dask
import dask.array as da
import dask.dataframe as dd

# Data manipulation and analysis
import xarray as xr
import polars as pl

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import bokeh
import altair as alt

# Machine learning for science
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Specialized scientific libraries
import astropy
import networkx as nx

# Time series analysis
import statsmodels.api as sm

# Signal processing
import librosa

# Optimization and symbolic math
import cvxpy
import sympy

# Statistics and probability
import pymc3 as pm

# Image processing
import skimage

# Geospatial analysis
import geopandas as gpd

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn

# Utilities
import h5py
import zarr
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging


def analyze_data():
    """Simple analysis function."""
    # Generate sample data
    data = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100)
    })

    # Basic analysis
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    return {
        'pca_variance': pca.explained_variance_ratio_.tolist(),
        'data_shape': data.shape
    }


if __name__ == "__main__":
    result = analyze_data()
    print(f"Analysis complete: {result}")
''')

        return sci_project


def create_ml_frameworks_fixtures(base_path: Path) -> dict[str, Path]:
    """Create all ML framework fixtures."""
    fixture = MLFrameworksFixture(base_path)

    return {
        "tensorflow": fixture.create_tensorflow_project(),
        "pytorch": fixture.create_pytorch_project(),
        "huggingface": fixture.create_huggingface_project(),
        "scientific": fixture.create_scientific_computing_project(),
    }
