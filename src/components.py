from azure.identity import DefaultAzureCredential
from azure.ai import ml
from azure.ai.ml import entities
from azure.ai.ml import dsl
import mldesigner

# Authenticate
credential = DefaultAzureCredential()

# Get a handle to workspace
ml_client = ml.MLClient.from_config(credential=credential)

# Print workspace details to confirm
print(f"Connected to workspace: {ml_client.workspace_name}")

# Get already made enviroment
pipeline_job_env = ml_client.environments.get(name="nnUNet", version="5")
print(f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}")

# Create data convert component
convert_msd_dataset = ml.command(
    name="convert_msd_dataset",
    display_name="Convert MSD dataset",
    description="Converts MSD dataset to nnunet dataset",
    inputs=dict(
        msd_dataset_folder=ml.Input(type="uri_folder"),
        num_processes=24,
    ),
    outputs=dict(
        nnunet_raw_folder=ml.Output(type="uri_folder", mode="rw_mount"),
    ),
    # The source folder of the component
    code="./convert_msd_dataset",
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
    command="""
df -H
python convert_msd_dataset.py -i ${{inputs.msd_dataset_folder}} -np ${{inputs.num_processes}} -o ${{outputs.nnunet_raw_folder}}/Dataset001_TEST
""",
)
# Now we register the component to the workspace
convert_msd_dataset_component = ml_client.create_or_update(convert_msd_dataset.component)

# Create (register) the component in your workspace
print(f"Component {convert_msd_dataset.name} with Version {convert_msd_dataset.version} is registered")

# Create ML component
train_nnunet = ml.command(
    name="train_nnunet",
    display_name="Train nnUNet",
    description="Trains a compatible dataset with nnUNet",
    inputs=dict(
        nnunet_raw_folder=ml.Input(type="uri_folder"),
        nnunet_n_proc=32,
        unet_configuration="3d_fullres",
        unet_fold=0,
    ),
    outputs=dict(
        nnunet_results_folder=ml.Output(type="uri_folder", mode="rw_mount"),
    ),
    # The source folder of the component
    code="./train_nnunet",
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
    command="""
echo ${{inputs.nnunet_n_proc}}
export nnUNet_n_proc_DA="${{inputs.nnunet_n_proc}}"
pwd
echo ${{inputs.nnunet_raw_folder}}
ls -la ${{inputs.nnunet_raw_folder}}
ls -la ${{inputs.nnunet_raw_folder}}/Dataset001_TEST
export nnUNet_raw="${{inputs.nnunet_raw_folder}}"
mkdir /mnt/nnunet_preprocessed
export nnUNet_preprocessed="/mnt/nnunet_preprocessed"
export nnUNet_results="${{outputs.nnunet_results_folder}}"
echo "planning and preprocessing"
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
echo "training"
nnUNetv2_train 1 ${{inputs.unet_configuration}} ${{inputs.unet_fold}} """,
)

# Now we register the component to the workspace
train_nnunet_component = ml_client.create_or_update(train_nnunet.component)

# Create (register) the component in your workspace
print(f"Component {train_nnunet.name} with Version {train_nnunet.version} is registered")
