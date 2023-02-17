#!/usr/bin/env python
#
# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import logging
import os
import json
os.environ['TRANSFORMERS_CACHE'] = '/tmp/hf/cache'
os.environ["HF_HOME"] = '/tmp/hf'

import subprocess
import sys
import shutil


if not os.path.exists("/tmp/diffusers/"):
    shutil.copytree("diffusers/", "/tmp/diffusers/")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "/tmp/diffusers/"])


import torch
from diffusers import DDIMScheduler
from diffusers.pipelines import StableDiffusionPipeline
import safetensors as st


# import deepspeed
from djl_python.inputs import Input
from djl_python.outputs import Output
from typing import Optional
from io import BytesIO
import base64



def get_torch_dtype_from_str(dtype: str):
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype is None:
        return None
    raise ValueError(f"Invalid data type: {dtype}")

def encode_images(images):
    encoded_images = []
    for image in images:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue())
        encoded_images.append(img_str.decode("utf8"))
    
    return encoded_images


class StableDiffusionService(object):

    def __init__(self):
        self.pipeline = None
        self.initialized = False
        self.ds_config = None
        self.logger = logging.getLogger()
        self.model_id_or_path = None
        self.data_type = None
        self.device = None
        self.world_size = None
        self.max_tokens = None
        self.tensor_parallel_degree = None
        self.save_image_dir = None

    def initialize(self, properties: dict):
        # If option.s3url is used, the directory is stored in model_id
        # If option.s3url is not used but model_id is present, we download from hub
        # Otherwise we assume model artifacts are in the model_dir
        self.model_id_or_path = properties.get("model_id") or properties.get("model_dir")
        self.data_type = get_torch_dtype_from_str(properties.get("dtype"))
        self.max_tokens = int(properties.get("max_tokens", "1024"))
        self.device = int(os.getenv("LOCAL_RANK", "0"))
        self.tensor_parallel_degree = int(
            properties.get("tensor_parallel_degree", 1))
        self.ds_config = self._get_ds_config_for_dtype(self.data_type)

        if os.path.exists(self.model_id_or_path):
            config_file = os.path.join(self.model_id_or_path, "model_index.json")
            if not os.path.exists(config_file):
                raise ValueError(
                    f"{self.model_id_or_path} does not contain a model_index.json."
                    f"This is required for loading stable diffusion models from local storage"
                )

        kwargs = {}
        if self.data_type == torch.float16:
            kwargs["torch_dtype"] = torch.float16
            kwargs["revision"] = "fp16"

        pipeline = StableDiffusionPipeline.from_pretrained(self.model_id_or_path, **kwargs)
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        
        pipeline.to(f"cuda:{self.device}")
        pipeline.unet.enable_xformers_memory_efficient_attention()
        
        # deepspeed.init_distributed()
        # engine = deepspeed.init_inference(getattr(pipeline, "model", pipeline),
        #                                   **self.ds_config)

        # if hasattr(pipeline, "model"):
        #     pipeline.model = engine

        self.pipeline = pipeline
        self.initialized = True

    def _get_ds_config_for_dtype(self, dtype):
        # This is a workaround due to 2 issues with DeepSpeed 0.7.5
        # 1. No kernel injection is available for stable diffusion using fp32 (kernels only written for fp16)
        # 2. Changes in our bf16 fork raise an error, but the original deepspeed codebase defaults to fp16
        #    when dtype is not set explicitly. We need to be explicit here with this config
        ds_config = {
            # TODO: Figure out why cuda graph doesn't work for stable diffusion via DS
            "enable_cuda_graph": False,
            "dtype": dtype,
            "mp_size": self.tensor_parallel_degree
        }
        if dtype == torch.float16:
            ds_config["replace_with_kernel_inject"] = True
            ds_config["replace_method"] = "auto"
        else:
            ds_config["replace_with_kernel_inject"] = False
            ds_config["replace_method"] = None
        return ds_config

    def inference(self, inputs: Input):
        try:
            content_type = inputs.get_property("Content-Type")
            if content_type == "application/json":
                request = inputs.get_as_json()
                prompt = request.pop("prompt")
                params = request.pop("parameters")
                
                if "latents" in params:
                    params["latents"] = st.torch.load(base64.b64decode(params["latents"].encode("utf8")))["latents"].to(f"cuda:{self.device}")
                
                images, latents, step = self.pipeline(prompt, **params)
                
                
                
            else:
                raise Exception("unsupported content type. Use application/json")
                # prompt = inputs.get_as_string()
                # result = self.pipeline(prompt)
            
            
            encoded_images = encode_images(images)
            latents_st = st.torch.save({"latents":latents})
            b64_latents = base64.b64encode(latents_st).decode("utf8")
            
            response = dict(images=encoded_images, latents=b64_latents, step=step)
            json_resp = json.dumps(response)
            
            
            outputs = Output().add(json_resp).add_property(
                "content-type", "application/json")

        except Exception as e:
            logging.exception("DeepSpeed inference failed")
            outputs = Output().error(str(e))
        return outputs


_service = StableDiffusionService()


def handle(inputs: Input) -> Optional[Output]:
    if not _service.initialized:
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        return None

    return _service.inference(inputs)