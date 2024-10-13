import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, ProcessingStep, CacheConfig
from sagemaker.workflow.pipeline import Pipeline
import boto3

# Step 1: Create an IAM role for SageMaker
# Go to the IAM console: https://console.aws.amazon.com/iam/
# Click on "Roles" and then "Create role".
# Select "SageMaker" as the service, and attach the necessary policies like "AmazonS3FullAccess" and "AmazonSageMakerFullAccess".
# Name the role as "SageMakerExecutionRole" and click "Create role".
role = 'arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole'

# Step 2: Create a Docker image or use a pre-built one
# Option 1: Use a built-in Amazon SageMaker image URI for scikit-learn
from sagemaker.image_uris import retrieve
image_uri = retrieve(framework='sklearn', region='us-east-1', version='0.23-1')
# Option 2: Create your own custom Docker image and push it to Amazon ECR
# Follow AWS ECR instructions to create and push a Docker image: https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html

session = sagemaker.Session()
bucket = 'mlops-pipeline-artifacts'

# Data preprocessing step
sklearn_processor = SKLearnProcessor(framework_version='0.23-1',
                                     role=role,
                                     instance_type='ml.m5.large',
                                     instance_count=1)
step_process = ProcessingStep(
    name='PreprocessingStep',
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(source=f's3://{bucket}/raw-data', destination='/opt/ml/processing/input')
    ],
    outputs=[
        ProcessingOutput(output_name='train', source='/opt/ml/processing/train', destination=f's3://{bucket}/train-data'),
        ProcessingOutput(output_name='test', source='/opt/ml/processing/test', destination=f's3://{bucket}/test-data')
    ],
    code='preprocessing.py'
)

# Model training step
estimator = Estimator(
    image_uri=image_uri,  # Use the retrieved or custom image URI
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://{bucket}/output'
)
step_train = TrainingStep(
    name='TrainingStep',
    estimator=estimator,
    inputs={'train': step_process.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri}
)

# Create and run pipeline
pipeline = Pipeline(
    name='IrisTrainingPipeline',
    steps=[step_process, step_train],
    sagemaker_session=session,
)

pipeline.upsert(role_arn=role)
