#!/usr/bin/env bash

# This script builds Docker container and pushes it to ECR, 
# so it is available for training and inference in Amazon Sagemaker.

# Script takes 3 arguments:
#    - image - required, this is container name which will be used when building locally and pushing to Amazon ECR;
#    - tag - optional, if provided, it will be used as ":tag" of your container name; otherwise, ":latest" will be used;
#    - dockerfile - optional, if provided, then docker will build container using specific dockerfile (e.g. "Dockerfile.serving"); otherwise, default "Dockerfile" will be used.

# Usage examples:
#    1. "./build_and_push.sh d2-sm-coco-serving debug Dockerfile.serving"
#    2. "./build_and_push.sh d2-sm-coco v2"

image=$1
tag=$2
dockerfile=$3

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi


# Get the region defined in the current configuration (default to us-east-2 if none defined)
region=$(aws configure get region)
region=${region:-us-east-2}

if [ "$tag" == "" ]
then
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
else
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${tag}"
fi

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

if [ "$dockerfile" == "" ]
then
    docker build  -t ${image} .
else
    docker build -t ${image} . -f ${dockerfile}
fi

docker tag ${image} ${fullname}
docker push ${fullname}
