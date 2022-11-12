import os

import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy)


def init_components(
    data_dir,
    transform_module,
    training_module,
    training_steps,
    eval_steps,
    serving_model,
):

    output = example_gen_pb2.Output(
        split_output = example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train',hash_buckets=8)
        ])
    )

    example_gen = CsvExampleGen(
        input_base=data_dir,
        output_config=output
    )





    return components