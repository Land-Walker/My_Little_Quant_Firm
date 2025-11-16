# file: pts/model/time_grad/time_grad_estimator.py

from typing import List, Optional

import torch

from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import TimeFeature
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import copy_parameters
from gluonts.model.predictor import Predictor
from gluonts.transform import (
    Transformation,
    Chain,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    ValidationSplitSampler,
    TestSplitSampler,
    RenameFields,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
)

from pts.trainer import Trainer
from pts.feature import (
    fourier_time_features_from_frequency,
    lags_for_fourier_time_features_from_frequency,
)
from pts.model import PyTorchEstimator
from pts.model.utils import get_module_forward_input_names

# 🔥 NEW: explicit-conditioning model
from .time_grad_network import (
    TimeGradTrainingNetwork,
    TimeGradPredictionNetwork,
)


class TimeGradEstimator(PyTorchEstimator):
    """
    Patches:
    - Dynamically infer conditioning dims:
        dyn_dim  = FEAT_DYNAMIC_REAL last dimension (if used)
        static_dim = FEAT_STATIC_REAL dimension (if used)
    - conditioning_length computed automatically
      = number of channels sent to EpsilonTheta cond_upsampler
      = dyn_dim + static_dim
    """

    def __init__(
        self,
        input_size: int,
        freq: str,
        prediction_length: int,
        target_dim: int,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "LSTM",
        num_parallel_samples: int = 100,
        dropout_rate: float = 0.1,
        cardinality: List[int] = [1],
        embedding_dimension: int = 5,
        diff_steps: int = 100,
        loss_type: str = "l2",
        beta_end=0.1,
        beta_schedule="linear",
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        scaling: bool = True,
        pick_incomplete: bool = False,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,

        # NEW flags
        use_feat_dynamic_real: bool = False,
        use_feat_static_real: bool = False,

        **kwargs,
    ) -> None:

        super().__init__(trainer=trainer, **kwargs)

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )

        self.input_size = input_size
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.num_parallel_samples = num_parallel_samples
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension

        self.diff_steps = diff_steps
        self.loss_type = loss_type
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.residual_layers = residual_layers
        self.residual_channels = residual_channels
        self.dilation_cycle_length = dilation_cycle_length

        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_feat_static_real = use_feat_static_real

        # automatically determine lags
        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else lags_for_fourier_time_features_from_frequency(freq_str=freq)
        )

        # automatically determine time features
        self.time_features = (
            time_features
            if time_features is not None
            else fourier_time_features_from_frequency(self.freq)
        )

        self.history_length = self.context_length + max(self.lags_seq)
        self.pick_incomplete = pick_incomplete
        self.scaling = scaling

        # Samplers
        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )
        self.validation_sampler = ValidationSplitSampler(
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )

        # These will be filled from actual batch during transform
        self.dyn_dim = None
        self.static_dim = None
        self.conditioning_length = None  # dyn_dim + static_dim


    ###########################################################################
    # TRANSFORMATION — also auto-detect dims
    ###########################################################################
    def create_transformation(self) -> Transformation:

        vstack_inputs = [FieldName.FEAT_TIME]

        if self.use_feat_dynamic_real:
            vstack_inputs.append(FieldName.FEAT_DYNAMIC_REAL)

        transforms = [
            AsNumpyArray(field=FieldName.TARGET, expected_ndim=2),
            ExpandDimArray(field=FieldName.TARGET, axis=None),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=self.time_features,
                pred_length=self.prediction_length,
            ),
        ]

        # STATIC REAL
        if not self.use_feat_static_real:
            transforms.append(SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_REAL, value=[0.0]))
        else:
            transforms.append(AsNumpyArray(field=FieldName.FEAT_STATIC_REAL, expected_ndim=1))

        # STACK TIME + DYNAMIC FEATURES
        transforms.append(
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=vstack_inputs,
            )
        )

        transforms.append(
            TargetDimIndicator(
                field_name="target_dimension_indicator",
                target_field=FieldName.TARGET,
            )
        )

        # Always carry static_cat
        transforms.append(AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1))

        return Chain(transforms)


    ###########################################################################
    # INSTANCE SPLITTER
    ###########################################################################
    def create_instance_splitter(self, mode: str):

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.history_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
                FieldName.FEAT_STATIC_REAL,
            ],
            dummy_value=0.0,
        ) + RenameFields({
            f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
            f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
        })


    ###########################################################################
    # NETWORK BUILDING — determine conditioning dims from a real batch
    ###########################################################################
    def _infer_condition_dims(self, batch):
        """
        Auto-detect dyn_dim and static_dim using one minibatch.
        Called inside create_training_network.
        """

        # static feat shape: (B, static_dim)
        if self.use_feat_static_real:
            static_feat = batch["feat_static_real"]    # gluonts field
            self.static_dim = static_feat.shape[-1]
        else:
            self.static_dim = 0

        # dynamic conditioning = FEAT_TIME minus original FEAT_TIME components
        if self.use_feat_dynamic_real:
            feat_time = batch["past_feat_time"]
            # feat_time shape: (B, past_length, time_feat + dyn_feat)
            # time_feat dim = len(self.time_features)
            total = feat_time.shape[-1]
            time_dim = len(self.time_features)
            self.dyn_dim = total - time_dim
        else:
            self.dyn_dim = 0

        # total conditioning vector that goes to EpsilonTheta
        self.conditioning_length = self.dyn_dim + self.static_dim


    ###########################################################################
    # CREATE TRAINING NETWORK
    ###########################################################################
    def create_training_network(self, device: torch.device):

        # get a batch to infer conditioning dims
        sample_batch = next(iter(self._train_iter))
        self._infer_condition_dims(sample_batch)

        return TimeGradTrainingNetwork(
            input_size=self.input_size,
            target_dim=self.target_dim,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            diff_steps=self.diff_steps,
            loss_type=self.loss_type,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            residual_layers=self.residual_layers,
            residual_channels=self.residual_channels,
            dilation_cycle_length=self.dilation_cycle_length,
            lags_seq=self.lags_seq,
            scaling=self.scaling,

            # 🔥 NEW
            dyn_dim=self.dyn_dim,
            static_dim=self.static_dim,
            conditioning_length=self.conditioning_length,
        ).to(device)


    ###########################################################################
    # CREATE PREDICTOR
    ###########################################################################
    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: TimeGradTrainingNetwork,
        device: torch.device,
    ) -> Predictor:

        prediction_network = TimeGradPredictionNetwork(
            input_size=self.input_size,
            target_dim=self.target_dim,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            diff_steps=self.diff_steps,
            loss_type=self.loss_type,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            residual_layers=self.residual_layers,
            residual_channels=self.residual_channels,
            dilation_cycle_length=self.dilation_cycle_length,
            lags_seq=self.lags_seq,
            scaling=self.scaling,

            # 🔥 SAME conditioning dims
            dyn_dim=self.dyn_dim,
            static_dim=self.static_dim,
            conditioning_length=self.conditioning_length,
            num_parallel_samples=self.num_parallel_samples,
        ).to(device)

        copy_parameters(trained_network, prediction_network)

        input_names = get_module_forward_input_names(prediction_network)
        prediction_splitter = self.create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
        )
