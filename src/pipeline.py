
"""
Make sure your cwd is in the directroy of this file
"""

from azure.identity import DefaultAzureCredential
from azure.ai import ml
from azure.ai.ml import entities
from azure.ai.ml import dsl
import mldesigner

import components

# Authenticate
credential = DefaultAzureCredential()

# Get a handle to workspace
ml_client = ml.MLClient.from_config(credential=credential)

# Print workspace details to confirm
print(f"Connected to workspace: {ml_client.workspace_name}")

# Het a handle of the data asset and print the URI
msd_dataset = ml_client.data.get(name="Task01-BrainTumour-uploaded", version="1")
print(f"Data asset URI: {msd_dataset.path}")

# Create pipeline from components
@dsl.pipeline(
    compute="gpu-cluster",  # "serverless" value runs pipeline on serverless compute
    description="test2-nnunet-pipeline",
)
def test2_pipeline(
    pipeline_job_msd_dataset_folder,
    pipeline_job_num_processes,
    pipeline_job_nnunet_n_proc,
    pipeline_job_unet_configuration,
    pipeline_job_unet_fold,
):
    print(f"MSD Dataset Folder: {pipeline_job_msd_dataset_folder}")
    print(f"Number of Processes: {pipeline_job_num_processes}")
    print(f"nnUNet Number of Processes: {pipeline_job_nnunet_n_proc}")
    print(f"UNet Configuration: {pipeline_job_unet_configuration}")
    print(f"UNet Fold: {pipeline_job_unet_fold}")

    # using convert_msd_dataset like a python call with its own inputs
    convert_msd_dataset_job = components.convert_msd_dataset(
        msd_dataset_folder=pipeline_job_msd_dataset_folder,
        num_processes=pipeline_job_num_processes,
    )

    print(f"Convert outputs: {convert_msd_dataset_job.outputs}")
    print(f"UNet RAW folder: {convert_msd_dataset_job.outputs.nnunet_raw_folder.path}")

    # train job
    train_nnunet_job = components.train_nnunet(
        nnunet_raw_folder=convert_msd_dataset_job.outputs.nnunet_raw_folder,  # note: using outputs from previous step
        nnunet_n_proc=pipeline_job_nnunet_n_proc, 
        unet_configuration=pipeline_job_unet_configuration, 
        unet_fold=pipeline_job_unet_fold, 
    )

    # a pipeline returns a dictionary of outputs
    # keys will code for the pipeline output identifier
    return {
        "nnunet_results_folder": train_nnunet_job.outputs.nnunet_results_folder,
    }


# Let's instantiate the pipeline with the parameters of our choice
pipeline = test2_pipeline(
    pipeline_job_msd_dataset_folder=ml.Input(type="uri_folder", path=msd_dataset.path),
    pipeline_job_num_processes=24,
    pipeline_job_nnunet_n_proc=32, # 32 for nvidia A100
    pipeline_job_unet_configuration="3d_fullres",
    pipeline_job_unet_fold=0,
)

# submit the pipeline job
pipeline_job = ml_client.jobs.create_or_update(
    pipeline,
    experiment_name="task01_braintumour", # Project's name
)
ml_client.jobs.stream(pipeline_job.name)
