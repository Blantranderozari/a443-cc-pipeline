import os
import sys

from typing import Text

from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

PIPELINE_NAME = "customer-churn-pipeline"

# pipeline inputs
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/customer_churn_transform.py"
TRAINER_MODULE_FILE = "modules/customer_churn_trainer.py"
# requirement_file = os.path.join(root,"requirements.txt")

# pipeline outputs
OUTPUT_BASE = "output"
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")

def init_local_pipeline(
    components, pipeline_root: Text
) -> pipeline.Pipeline:

    logging_info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing"
        # 0 auto-detect based on the number of CPUs available
        # during execution
        "----direct_num_workers=0"
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        eam_pipeline_args=beam_args
    )