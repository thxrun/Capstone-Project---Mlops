<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
 
</head>
<body>

<h1>Capstone Machine Learning Pipeline with MLOps & EKS</h1>

<p>
This project implements an end‑to‑end <strong>MLOps</strong> workflow for a machine learning system, starting from project scaffolding and data engineering, through model training with MLflow and DVC, to containerized deployment on Amazon EKS with CI/CD and monitoring via Prometheus and Grafana.
</p>

<h2>Overview</h2>
<p>
The repository demonstrates how to structure a production‑grade Python ML project with modular components, environment management, artifact pipelines, experiment tracking, and cloud deployment. It uses:
</p>
<ul>
  <li>Dagshub and MLflow for experiment tracking and model registry.</li>
  <li>DVC for reproducible data and model pipelines.</li>
  <li>AWS services (S3, EC2, ECR, IAM, EKS) for model storage and containerized deployment.</li>
  <li>Docker plus GitHub Actions for automated continuous integration and delivery.</li>
  <li>Prometheus and Grafana for monitoring the deployed service.</li>
</ul>

<h2>Key Features</h2>
<ul>
  <li><strong>Project scaffolding</strong>: Auto-generated template using <code>cookiecutter-data-science</code> with a structured <code>src</code> layout and Python modules.</li>
  <li><strong>Environment &amp; dependencies</strong>: Reproducible environment using a dedicated Conda environment (<code>atlas</code>) and pinned dependencies via <code>requirements.txt</code>.</li>
  <li><strong>Experiment tracking</strong>: MLflow integrated with Dagshub for centralized logging of metrics, parameters, and model artifacts.</li>
  <li><strong>Data &amp; model pipelines</strong>: DVC‑driven pipeline for data ingestion, preprocessing, feature engineering, model training, and evaluation, including a local S3‑style cache and later AWS S3 as remote.</li>
  <li><strong>Custom logging &amp; exception layer</strong>: Centralized logger and domain‑specific exception classes used across the ML pipeline for observability.</li>
  <li><strong>Model registry on S3</strong>: Versioned models stored in an S3 bucket with configuration via <code>params.yaml</code> and DVC.</li>
  <li><strong>REST API via Flask</strong>: <code>flask_app/app.py</code> exposes HTTP endpoints for model prediction, with environment‑driven configuration for secrets and credentials.</li>
  <li><strong>Docker &amp; CI/CD</strong>: Dockerized Flask app, GitHub Actions pipeline, and ECR support for automated build, push, and deployment.</li>
  <li><strong>Deployment on EKS</strong>: Kubernetes manifests and GitHub CI/CD deploy the service on an EKS cluster via ECR images and LoadBalancer.</li>
  <li><strong>Monitoring</strong>: Prometheus and Grafana servers on EC2 monitor the Flask app endpoint and expose metrics dashboards.</li>
</ul>

<h2>Tech Stack</h2>
<ul>
  <li><strong>Domain</strong>: Generic machine learning project (adaptable to insurance, health, etc.) with modular pipeline components.</li>
  <li><strong>Language</strong>: Python 3.10</li>
  <li><strong>Data / ML</strong>: Python ML stack (e.g., pandas, scikit‑learn, MLflow, DVC, PyYAML, etc. – defined in <code>requirements.txt</code>).</li>
  <li><strong>Experiment tracking</strong>: MLflow with Dagshub backend.</li>
  <li><strong>Data pipelines</strong>: DVC with local and S3 remote storage.</li>
  <li><strong>Cloud</strong>: AWS (S3, ECR, EC2, IAM, EKS, CloudFormation).</li>
  <li><strong>CI/CD</strong>: GitHub Actions with optional self‑hosted runner on EC2.</li>
  <li><strong>Containerization</strong>: Docker images stored in Amazon ECR.</li>
  <li><strong>Orchestration</strong>: Kubernetes (EKS) with <code>kubectl</code> and <code>eksctl</code>.</li>
  <li><strong>Monitoring</strong>: Prometheus (v2.46.0) and Grafana (v10.1.5) hosted on EC2 instances.</li>
  <li><strong>Web framework</strong>: Flask‑based prediction API in <code>flask_app/app.py</code>.</li>
</ul>

<h2>Repository Structure</h2>
<p>
The project follows a clean, MLOps‑oriented layout inspired by <code>cookiecutter‑data‑science</code>, with additional modules for logging, exceptions, and deployment.
</p>

<pre>
project-root/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   └── exp_notebooks.ipynb
├── models/
├── src/
│   ├── logger.py
│   ├── exception.py
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   ├── model_evaluation.py
│   └── register_model.py
├── flask_app/
│   ├── app.py
│   ├── templates/
│   └── static/
├── tests/
├── scripts/
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── .github/
    └── workflows/
        └── ci.yaml
</pre>

<h2>Local Setup &amp; Installation</h2>

<h3>Project Scaffolding</h3>
<ol>
  <li>Create a GitHub repo and clone it locally:
    <pre>git clone &lt;your-repo-url&gt;</pre>
  </li>
  <li>Create and activate a Conda environment:
    <pre>
conda create -n atlas python=3.10
conda activate atlas
    </pre>
  </li>
  <li>Install and run cookiecutter:
    <pre>
pip install cookiecutter
cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
mv src.models src.model
    </pre>
  </li>
  <li>Commit the initial structure:
    <pre>
git add . && git commit -m "Initial project structure from cookiecutter-data-science"
git push
    </pre>
  </li>
</ol>

<h3>MLflow &amp; Dagshub Setup</h3>
<ol>
  <li>Go to your Dagshub dashboard and connect the GitHub repo.</li>
  <li>Copy the experiment tracking URL and MLflow code snippet.</li>
  <li>Install required packages:
    <pre>pip install dagshub mlflow</pre>
  </li>
  <li>Run experiment notebooks inside <code>notebooks/</code> and push changes:
    <pre>git add . && git commit -m "Run MLflow experiments" && git push</pre>
  </li>
</ol>

<h3>DVC Pipeline Initialization</h3>
<ol>
  <li>Initialize DVC:
    <pre>dvc init</pre>
  </li>
  <li>Create a temporary local “S3” folder:
    <pre>mkdir local_s3</pre>
  </li>
  <li>Add a local DVC remote:
    <pre>dvc remote add -d mylocal local_s3</pre>
  </li>
  <li>Create <code>dvc.yaml</code> and <code>params.yaml</code> defining the pipeline steps (until model evaluation).</li>
  <li>Reproduce the pipeline:
    <pre>dvc repro</pre>
  </li>
  <li>Check status and commit:
    <pre>dvc status</pre>
  </li>
</ol>

<h3>Adding S3 as Remote Storage</h3>
<ol>
  <li>Create an IAM user with S3 access and note credentials.</li>
  <li>Create an S3 bucket (for example, <code>my-model-mlopsproj</code>) in <code>us-east-1</code>.</li>
  <li>Install DVC S3 and AWS CLI:
    <pre>pip install -U dvc[s3] awscli</pre>
  </li>
  <li>Configure AWS credentials:
    <pre>aws configure</pre>
  </li>
  <li>Optionally check and remove existing remotes:
    <pre>dvc remote list</pre>
    <pre>dvc remote remove &lt;name&gt;</pre>
  </li>
  <li>Add S3 as default remote:
    <pre>dvc remote add -d myremote s3://&lt;your-bucket-name&gt;</pre>
  </li>
</ol>

<h3>Logging, Exceptions &amp; Tests</h3>
<ol>
  <li>Implement a central logger in <code>src/logger.py</code> and test it in a demo script.</li>
  <li>Implement custom exception classes in <code>src/exception.py</code> and integrate them into data and model components.</li>
  <li>Create a <code>tests/</code> directory with unit and integration tests for the pipeline components.</li>
  <li>Ensure tests are runnable via a simple command:
    <pre>python -m pytest tests/</pre>
  </li>
</ol>

<h2>ML Pipeline Components</h2>

<h3>Data Ingestion, Preprocessing, and Feature Engineering</h3>
<ol>
  <li>In <code>src/data_ingestion.py</code>, implement functions to load raw data from files or external sources.</li>
  <li>In <code>src/data_preprocessing.py</code>, clean and normalize the data, handling missing values and outliers.</li>
  <li>In <code>src/feature_engineering.py</code>, create domain‑relevant features and save them to the <code>data/</code> layer.</li>
  <li>Ensure these steps are declared in <code>dvc.yaml</code> so they run as part of the pipeline.</li>
</ol>

<h3>Model Building &amp; Evaluation</h3>
<ol>
  <li>In <code>src/model_building.py</code>, implement model training and hyperparameter handling configured via <code>params.yaml</code>.</li>
  <li>In <code>src/model_evaluation.py</code>, compute metrics and log them to MLflow using the Dagshub tracking URL.</li>
  <li>In <code>src/register_model.py</code>, register the best model in MLflow and optionally push metadata to S3.</li>
</ol>

<h2>Model Registry, AWS Integration &amp; Flask API</h2>

<h3>Model Registry on S3</h3>
<ol>
  <li>Define model registry parameters in <code>params.yaml</code>, for example:
    <pre>
model_registry:
  bucket_name: "my-model-mlopsproj"
  key_prefix: "model-registry"
    </pre>
  </li>
  <li>Implement helpers in <code>src/</code> or a dedicated module to push and pull models from S3 using the AWS SDK.</li>
  <li>Integrate model registry calls into the DVC pipeline so that only models passing a threshold are promoted.</li>
</ol>

<h3>Flask Prediction API</h3>
<ol>
  <li>Create a <code>flask_app/</code> directory and add:
    <pre>flask_app/app.py</pre>
  </li>
  <li>Implement:
    <ul>
      <li>A root or <code>/predict</code> route to load the latest model from S3 or local artifacts and return predictions.</li>
      <li>A <code>/training</code> route (optional) to trigger retraining via the pipeline.</li>
    </ul>
  </li>
  <li>Add <code>templates/</code> and <code>static/</code> folders for HTML templates and CSS/JS assets.</li>
  <li>Run the app locally:
    <pre>cd flask_app && python app.py</pre>
  </li>
</ol>

<h2>Dockerization &amp; CI/CD with GitHub Actions</h2>

<h3>Docker Setup</h3>
<ol>
  <li>Install Docker on your machine and ensure Docker Desktop is running.</li>
  <li>In the root directory, create a Dockerfile and <code>.dockerignore</code> to containerize the Flask app.</li>
  <li>Copy the Flask app and required dependencies into the image and expose the correct port (e.g., 5000).</li>
  <li>From the root directory, build the image:
    <pre>docker build -t capstone-app:latest .</pre>
  </li>
  <li>Run the image:
    <pre>docker run -p 8888:5000 capstone-app:latest</pre>
  </li>
  <li>Pass the Dagshub token via environment variable:
    <pre>docker run -p 8888:5000 -e CAPSTONE_TEST=54b1d67648a9b1267ef906fsdfsd8b292f779f0 capstone-app:latest</pre>
  </li>
</ol>

<h3>GitHub Secrets &amp; CI/CD Pipeline</h3>
<ol>
  <li>Create GitHub repository secrets:
    <ul>
      <li><code>CAPSTONE_TEST</code>: Dagshub authentication token.</li>
      <li><code>AWS_ACCESS_KEY_ID</code>, <code>AWS_SECRET_ACCESS_KEY</code></li>
      <li><code>AWS_REGION</code> (e.g., <code>us-east-1</code>)</li>
      <li><code>ECR_REPOSITORY</code>: ECR repository URI (e.g., <code>&lt;account-id&gt;.dkr.ecr.us-east-1.amazonaws.com/capstone-proj</code>)</li>
      <li><code>AWS_ACCOUNT_ID</code></li>
    </ul>
  </li>
  <li>In <code>.github/workflows/ci.yaml</code>, define a workflow that:
    <ul>
      <li>Checks out the code.</li>
      <li>Runs tests in <code>tests/</code>.</li>
      <li>Builds the Docker image.</li>
      <li>Authenticates with ECR and pushes the image.</li>
      <li>Optionally deploys to EKS or an EC2 instance via <code>kubectl</code> or SSH.</li>
    </ul>
  </li>
  <li>Configure an IAM user with <code>AmazonEC2ContainerRegistryFullAccess</code> for CI/CD operations.</li>
</ol>

<h2>AWS EKS Setup &amp; Deployment</h2>

<h3>Prerequisites on Windows (PowerShell)</h3>
<p>
If you are using Windows and AWS CLI from Anaconda, clean up the conflicting installation:
</p>
<ol>
  <li>Check the AWS CLI path:
    <pre>Get-Command aws</pre>
  </li>
  <li>If it points to Anaconda, uninstall the Python‑based AWS CLI:
    <pre>pip uninstall awscli</pre>
  </li>
  <li>Install AWS CLI v2 from the official MSI and add <code>C:\Program Files\Amazon\AWSCLIV2\</code> to your <code>PATH</code>.</li>
  <li>Verify:
    <pre>aws --version</pre>
  </li>
</ol>

<h3>Install kubectl &amp; eksctl</h3>
<ol>
  <li>Download and install <code>kubectl</code>:
    <pre>
Invoke-WebRequest -Uri "https://dl.k8s.io/release/v1.28.2/bin/windows/amd64/kubectl.exe" -OutFile "kubectl.exe"
Move-Item -Path .\kubectl.exe -Destination "C:\Windows\System32"
kubectl version --client
    </pre>
  </li>
  <li>Download and install <code>eksctl</code>:
    <pre>
Invoke-WebRequest -Uri "https://github.com/weaveworks/eksctl/releases/download/v0.158.0/eksctl_Windows_amd64.zip" -OutFile "eksctl.zip"
Expand-Archive -Path .\eksctl.zip -DestinationPath .
Move-Item -Path .\eksctl.exe -Destination "C:\Windows\System32\eksctl.exe"
eksctl version
    </pre>
  </li>
</ol>

<h3>Create EKS Cluster</h3>
<ol>
  <li>Create a managed EKS cluster:
    <pre>
eksctl create cluster \
  --name flask-app-cluster \
  --region us-east-1 \
  --nodegroup-name flask-app-nodes \
  --node-type t3.small \
  --nodes 1 \
  --nodes-min 1 \
  --nodes-max 1 \
  --managed
    </pre>
  </li>
  <li>Update the kubectl config:
    <pre>aws eks --region us-east-1 update-kubeconfig --name flask-app-cluster</pre>
    <li>Verify cluster connectivity:
    <pre>kubectl get nodes
kubectl get namespaces</pre>
  </li>
</ol>

<h3>Deploy Application to EKS</h3>
<ol>
  <li>Create a Kubernetes deployment manifest (<code>deployment.yaml</code>) that references your ECR image.</li>
  <li>Apply the deployment and service:
    <pre>kubectl apply -f deployment.yaml</pre>
  </li>
  <li>Retrieve the External IP of the LoadBalancer:
    <pre>kubectl get svc flask-app-service</pre>
  </li>
  <li>Test the live endpoint:
    <pre>curl http://&lt;external-ip&gt;:5000/predict</pre>
  </li>
</ol>

<h2>Monitoring Setup (Prometheus & Grafana)</h2>
<p>To ensure production reliability, we host monitoring tools on dedicated EC2 instances.</p>

<h3>Prometheus Configuration</h3>
<ol>
  <li>Launch an Ubuntu EC2 (t3.medium) and install Prometheus.</li>
  <li>Configure <code>/etc/prometheus/prometheus.yml</code> to scrape your EKS LoadBalancer:
    <pre>scrape_configs:
  - job_name: 'flask-app'
    static_configs:
      - targets: ['&lt;EKS-LB-External-IP&gt;:5000']</pre>
  </li>
  <li>Start Prometheus:
    <pre>prometheus --config.file=/etc/prometheus/prometheus.yml</pre>
  </li>
</ol>

<h3>Grafana Visualization</h3>
<ol>
  <li>Launch an Ubuntu EC2 and install Grafana via <code>apt</code>.</li>
  <li>Access the UI at <code>http://&lt;ec2-ip&gt;:3000</code>.</li>
  <li>Add Prometheus as a Data Source using the Prometheus EC2 internal/public IP.</li>
  <li>Build dashboards to track inference latency and system resource usage.</li>
</ol>

<h2>Resource Cleanup</h2>
<p><strong>Crucial:</strong> Delete AWS resources once testing is complete to avoid high billing.</p>
<pre># Delete K8s resources
kubectl delete deployment flask-app
kubectl delete service flask-app-service

# Delete the EKS Cluster (takes ~15 mins)
eksctl delete cluster --name flask-app-cluster --region us-east-1

# Manually delete ECR images, S3 buckets, and Monitoring EC2 instances via AWS Console.</pre>

<hr>
<p style="text-align: center;">Developed by <strong>Tharun Karthik G</strong> | MLOps & Python Backend Engineer</p>

</body>
</html>
