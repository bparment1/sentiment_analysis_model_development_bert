# Sentiment analysis Modeling

This project builds a sentiment analysis model using BERT with pre-Trained weights. The code is developed locally and deployed on the cloud using Google Cloud Computing platform. The following tools are leveraged:

- mlflow
- huggingface
- postgres sql
- Cloud SQL
- cloud storage
- google artifact registry
- docker
- cloud run
- github and git

The general structure of this project repo is as follow:

```
├── configs
│   ├── default.yml
│   ├── freeze_base.yml
│   └── full_finetune.yml
├── data    /store data
├── Dockerfile 
├── mlruns /mflow tracking folder
├── models /store model when training
│   └── checkpoints
│       ├── best_model_epoch1.pt
├── pyproject.toml
├── README.md
├── requirements.txt
├── sentiment_model /code packaging for the app
│   ├── __init__
│   ├── data.py
│   ├── evaluation.py
│   ├── model.py
│   └── training.py
├── train.py
├── evaluate.py
└── uv.lock
```

# Installing Environmnent and running training

Use uv to setup the environment:

```
uv sync
```

You can get help about the script:

```
python your_script.py --help
```

You can run the code from the command line:

```
python train.py \
--model_name bert-base-uncased \
--freeze_base \
--max_batches 10 \
--number_epoch 3 \
--experiment_name bert-train-sentiment \
--tracking_uri file:./mlruns 
```

# Connect to the database Cloudsql

If you have already setup the postgres database on google cloudsql you can connect via proxy:

```
cloud-sql-proxy $PROJECT_ID:$REGION:$INSTANCE_NAME &
```

Make sure first no other database is running on the localhost at port 5432:

```
ps aux | grep postgres
```

Stop it if want
```
sudo systemctl stop postgresql
```

Access the database via psql:

```
psql -h $DB_RE_HOST -U $DB_RE_USER -d DB_RE_DB_NAME -p 5432
```

Note that because we are using cloud-sql-proxy the hostname is the localhost 127.0.0.1.

# Setting up the cloud infrastructure

Below we show how set up the cloud infrastructure to track the modeling experiment. This involves setting up a postgres instance on GCP and a bucket cloud storage to track artifacts and experiments results in mlflow. The last steps shows how to package the code in a container to run as cloud run job.

Set the following env variable in your terminal before running the commands:

- INSTANCE_NAME: cloudsql instance name
- PROJECT_ID: google project ID
- REGION: region used in the project e.g. us-central1

You can get the project ID this way:
```
gcloud config get-value project
```

# Database setup: Cloudsql

Create a cloudsql instance of a postgres database using the GUI or command line:

```
gcloud sql instances create $INSTANCE_NAME \
  --database-version=POSTGRES_16 \
  --region=REGION \
  --availability-type=ZONAL \
  --storage-type=HDD \
  --storage-size=10GB \
  --assign-ip \
  --cpu=1 \
  --memory=3840MB
```

CLOUDSQL with gcloud command, with the following setup,
- postgres version: postgres 16, 
- region: us-east1
- avaibility: single zone, 
- storage type: HDD (cheaper but slower than SDD)
- storage capacity: 10 gb 
--assign-ip gives you the public IP. If you want to restrict who can connect to it you'll need to add authorized networks after creation
- cpu: one for the database (lowest and cheapest)
- memory: 3640 MB (lowest and cheapest)

Once setup we can access the database locally. To do so, we will use cloud proxy so that the database will appear on the localhost at port 5432.

First we need to install cloud sql proxy.

```
gcloud components install cloud-sql-proxy
```

Download the binary directly

```
curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.15.2/cloud-sql-proxy.linux.amd64

chmod +x cloud-sql-proxy
sudo mv cloud-sql-proxy /usr/local/bin/
```

Check if installed:

```
cloud-sql-proxy --version
```
Now we can list the cloudsql instances:

```
gcloud sql instances list
gcloud sql instances describe [INSTANCE_NAME] --project=[PROJECT_ID]
```

First authorize your ip on CLOUDSQL

```
MY_IP=$(curl -s https://ipinfo.io/ip)
gcloud sql instances patch INSTANCE_NAME --authorized-networks=$MY_IP
```
To obtain the public ip to connect:

```
gcloud sql instances describe INSTANCE_NAME --format="value(ipAddresses.ipAddress)"
```

To connect, we can use the the public ip directly, this is not as secure as using the proxy locally, but we show how to do it using a dummy ip:
psql -h <host> -p <port> -U <username> -d <database_name>

```
psql -h "38.11.402.481" -p 5432 -U "postgres" -d "postgres" -W
```

# Setup Postgres Cloudsql as Mlflow backend

Make sure you have the required pacakges (in particular cloud sql proxy).

First check that no local instance is running. If it is running either stop it or 
use another port when using cloud proxy:

```
ps aux | grep postgres
#stop it if want
sudo systemctl stop postgresql
```

Then run this one if using the default port (no postgres running locally):

```
cloud-sql-proxy $PROJECT_ID:$REGION:$INSTANCE_NAME &
psql -h 127.0.0.1 -p 5432 -U postgres -W
```

Change port if you have another instance of posgres runnning locally:

```
cloud-sql-proxy --port 5433 PROJECT_ID:REGION:INSTANCE_NAME &
psql -h 127.0.0.1 -p 5433 -U postgres
```

If we are using, port 5432 for the connection.

```
sudo systemctl stop postgresql
cloud-sql-proxy PROJECT_ID:REGION:INSTANCE_NAME &
psql -h 127.0.0.1 -p 5432 -U postgres -W
```
To start the instance:

```
sudo systemctl start postgresql
```

## Create a user and database 

We can either use cloudsql or directly postgres psql. If using Cloudsql use the following:

```
gcloud sql instances patch INSTANCE_NAME \
  --authorized-networks=YOUR_IP/32
```

Create a database:

```
gcloud sql databases create my-database --instance=INSTANCE_NAME
```

Create a user:

```
gcloud sql users create my-user \
  --instance=INSTANCE_NAME \
  --password=your-password
```

To use posgres as a data backend for mlflow we need a database and user. Let's create both and grant the necessary provileges and roles:

```
-- Create a database for mlflow

"CREATE DATABASE mlflow;"

-- Connect to your Cloud SQL instance
psql -h 127.0.0.1 -U postgres

-- list existing databases
\l

-- list existing users without and with details
\du
\du+

-- Create a dedicated user
CREATE USER mlflow_user WITH PASSWORD 'strong-password-here';

-- Grant access to the mlflow database
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow_user;

-- Connect to the mlflow database to grant schema permissions
\c mlflow

-- Grant schema permissions (important for Postgres 15+)
GRANT ALL ON SCHEMA public TO mlflow_user;

-- Instead of ALL PRIVILEGES, be more specific
GRANT CONNECT ON DATABASE mlflow TO mlflow_user;
GRANT USAGE ON SCHEMA public TO mlflow_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO mlflow_user;

-- So future tables created by migrations are also covered
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO mlflow_user;
```

Now we setup the cloudsql proxy and that the mlflow database and the user mlflow_user were created, you can login using the normal command as if it were a localhost database:

```
psql -h 127.0.0.1 -p 5432 -U mlflow_user -d mlflow -W
```

You will prompted for the password. If you prefer you can pass the command direclty in a env variable:

```
PGPASSWORD=$DB_RE_DB_NAME psql -h 127.0.0.1 -p 5432 -U mlflow_user -d mlflow
```

# Setup Cloud storage for mlflow

Next we need to set up a bucket to store the model and other artifacts to track the experiments in mlflow. This will require creating a new service account for mlflow:

BUCKET_NAME=mlflow-artifacts-sentiment-analysis-app

```
gcloud storage buckets create gs://$BUCKET_NAME \
  --location=$REGION
```
Create service account for mflow.

```
gcloud iam service-accounts create mlflow-sa \
  --display-name "MLflow Service Account"
```

Restrict Bucket Access (Recommended)

```
gcloud storage buckets add-iam-policy-binding gs://your-mlflow-artifacts \
  --member="serviceAccount:mlflow-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

This is more secure than granting project-wide access — limits the service account to only this bucket.

Grant Storage access

```
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:mlflow-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

# Download key (for local dev)

```
gcloud iam service-accounts keys create mlflow-sa-key.json \
  --iam-account=mlflow-sa@PROJECT_ID.iam.gserviceaccount.com
```

# Point your env to the key

```
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/mlflow-sa-key.json
```

## Starting MLFLOW server with cloud database and cloud storage after run

When everything is ready, you can test the script by launching locallly:

```
python train.py \
--model_name bert-base-uncased \
--freeze_base \
--max_batches 10 \
--number_epoch 3 \
--experiment_name bert-train-sentiment \
--tracking_uri "postgresql://$DB_RE_USER:$DB_RE_PASSWORD@$DB_RE_HOST:5432/$DB_RE_DB_NAME"
--artifact_location gs://mlflow-artifacts-sentiment-analysis-app
                         
```

After running you can launch the mlflow server

```
cloud-sql-proxy PROJECT_ID:REGION:YOUR_INSTANCE &

mlflow server \
  --backend-store-uri "postgresql://$DB_RE_USER:$DB_RE_PASSWORD@$DB_RE_HOST:5432/$DB_RE_DB_NAME" \
  --default-artifact-root gs://mlflow-artifacts-sentiment-analysis-app  \
  --host 127.0.0.1 \
  --port 5000
  ```

## Starting mflow for local run

If you are testing locally you would do the following:

run a local training job:

```
python train.py \
--model_name bert-base-uncased \
--freeze_base \
--max_batches 10 \
--number_epoch 3 \
--experiment_name bert-train-sentiment \
--tracking_uri file:./mlruns 
```

Then star mlflow server locally:

```
mlflow server \
  --backend-store-uri file:./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

Or you can use the simpler command:

```
mlflow ui --port 5000
```

# Run model evaluation

Using evaluate.py you can run a quick test with test data with 10 batches:

```
python evaluate.py \
  --checkpoint_path models/checkpoints/best_model_epoch5.pt \
  --max_batches 10
```

If you want to leverage mlflow for the evaluation using local mlrun you can do this:

```
python evaluate.py \
  --checkpoint_path models/checkpoints/best_model_epoch5.pt \
  --model_name bert-base-uncased \
  --experiment_name bert-train-sentiment
```

A few quick notes:

- mlflow does not provide automatic checkpoints for pyTorch pt but could use transfomers logmodel for mlflow.

- evaluation can be a separate job from training in production. 

# Building a docker image

docker run <image> --freeze_base --number_epoch 3

# Deploy with cloud run

# CI/CD githubaction