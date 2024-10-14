import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, ModelStep
from sagemaker.processing import ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.inputs import ProcessingInput, ProcessingOutput, TrainingInput

# Set up session
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::252927399996:role/SageMakerExecutionRole"  # Replace with your SageMaker role ARN

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
        ProcessingInput(source="s3://aish-mlops-bucket/raw_data/input.csv",  # Replace with actual data path
                        destination="/opt/ml/processing/input")
    ],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/train",
                         destination="s3://aish-mlops-bucket/processed_data/train"),
        ProcessingOutput(source="/opt/ml/processing/test",
                         destination="s3://aish-mlops-bucket/processed_data/test")
    ],
    code="preprocessing.py",
)

# Model Training Step
estimator = Estimator(
    image_uri="683313688378.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.2-2",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path="s3://aish-mlops-bucket/model_artifacts",  # Replace with your S3 bucket
)

training_step = TrainingStep(
    name="ModelTrainingStep",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data="s3://aish-mlops-bucket/processed_data/train",  # Use output from processing step
            content_type="text/csv"
        )
    }
)

# Model Registration Step
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
