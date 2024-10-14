import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, ModelStep
from sagemaker.processing import ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.inputs import TrainingInput

# Set up session
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"  # Replace with your SageMaker role ARN

# Data Pre-Processing Step
script_processor = ScriptProcessor(
    image_uri="683313688378.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
)

processing_step = ProcessingStep(
    name="DataPreprocessingStep",
    processor=script_processor,
    inputs=[
        # Add inputs
    ],
    outputs=[
        # Add outputs
    ],
    code="preprocessing.py",  # Add your data preprocessing script here
)

# Model Training Step
estimator = Estimator(
    image_uri="683313688378.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.2-2",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path="s3://YOUR_BUCKET_NAME/model_artifacts",  # Replace with your S3 bucket
)

training_step = TrainingStep(
    name="ModelTrainingStep",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data="s3://YOUR_BUCKET_NAME/train",  # Replace with your training data path
            content_type="text/csv"
        )
    }
)

# Model Evaluation and Registration Step
model = Model(
    image_uri="683313688378.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.2-2",
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role
)

model_step = ModelStep(
    name="RegisterModelStep",
    model=model,
    model_package_group_name="ModelPackageGroup",
    approval_status="PendingManualApproval"
)

# Create the SageMaker Pipeline
pipeline = Pipeline(
    name="MLModelPipeline",
    steps=[processing_step, training_step, model_step],
    sagemaker_session=sagemaker_session
)

# Execute the pipeline
pipeline.upsert(role_arn=role)
execution = pipeline.start()
print("Pipeline started. Execution ARN: ", execution.arn)
