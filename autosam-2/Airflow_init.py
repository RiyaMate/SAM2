"""
Airflow + AWS Orchestration Script for SAM2 Segmentation
Author: Riya Mate
 - Downloads image and model checkpoint from S3
 - Runs SAM2-based segmentation using PyTorch
 - Uploads result (mask visualization) back to S3
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import boto3
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import subprocess
from collections import OrderedDict

# ------------------ CONFIGURATION ------------------
AWS_REGION = "us-east-1"
S3_BUCKET = "your-s3-bucket-name"
S3_IMAGE_KEY = "inputs/sample_image.jpg"
S3_CHECKPOINT_KEY = "checkpoints/sam2.1_hiera_base_plus.pt"
S3_OUTPUT_KEY = "outputs/output.png"

LOCAL_IMAGE = "/tmp/input.jpg"
LOCAL_CHECKPOINT = "/tmp/sam2_checkpoint.pt"
LOCAL_OUTPUT = "/tmp/output.png"

HOME = "/usr/local/airflow/dags"  # MWAA/EC2 path where script runs
# ---------------------------------------------------


# ------------------ UTILITY FUNCTIONS ------------------
def download_from_s3(bucket, key, dest):
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.download_file(bucket, key, dest)
    print(f"✅ Downloaded s3://{bucket}/{key} → {dest}")


def upload_to_s3(bucket, key, src):
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(src, bucket, key)
    print(f"✅ Uploaded {src} → s3://{bucket}/{key}")


# ------------------ SEGMENTATION LOGIC ------------------
def run_segmentation(**kwargs):
    from autosam_utils import *
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.modeling import MaskDecoder, TwoWayTransformer, AutoSamSeg

    # Download required files
    download_from_s3(S3_BUCKET, S3_IMAGE_KEY, LOCAL_IMAGE)
    download_from_s3(S3_BUCKET, S3_CHECKPOINT_KEY, LOCAL_CHECKPOINT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_sam_seg_model(image_encoder, num_classes, checkpoint=None):
        prompt_embed_dim = 256
        sam_seg = AutoSamSeg(
            image_encoder=image_encoder,
            seg_decoder=MaskDecoder(
                num_multimask_outputs=1,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes=num_classes,
            ),
        )
        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location=device)
            loaded_keys = {k: v for k, v in state_dict.items() if k in sam_seg.state_dict()}
            sam_seg.load_state_dict(loaded_keys, strict=False)
            print("✅ Loaded pretrained weights")
        return sam_seg

    # Build model
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam2_model = build_sam2(model_cfg, LOCAL_CHECKPOINT, device=device)
    sam2_encoder = sam2_model.image_encoder.to(device)
    model = _build_sam_seg_model(sam2_encoder, num_classes=2).to(device)
    model.eval()

    # Load image
    img = Image.open(LOCAL_IMAGE).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        mask_pred, iou_pred = model(img_tensor)
        b, h, w = img_tensor.shape[0], img_tensor.shape[2], img_tensor.shape[3]
        mask_pred = mask_pred.view(b, -1, h, w)
        prob = torch.softmax(mask_pred, dim=1)
        pred_mask = (prob[:, 1] > 0.5).float()
        pred_mask_np = pred_mask.squeeze().cpu().numpy()

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask_np, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(LOCAL_OUTPUT)
    plt.close()
    print(f"✅ Output saved to {LOCAL_OUTPUT}")

    # Upload result to S3
    upload_to_s3(S3_BUCKET, S3_OUTPUT_KEY, LOCAL_OUTPUT)


# ------------------ AIRFLOW DAG DEFINITION ------------------
with DAG(
    dag_id="sam2_segmentation_orchestration",
    default_args={"owner": "riya.mate", "retries": 1},
    schedule_interval=None,  # Run manually
    start_date=days_ago(0),
    catchup=False,
    description="Orchestrates SAM2 segmentation pipeline on AWS using Airflow",
) as dag:

    run_job = PythonOperator(
        task_id="run_sam2_segmentation_pipeline",
        python_callable=run_segmentation,
        provide_context=True,
    )

