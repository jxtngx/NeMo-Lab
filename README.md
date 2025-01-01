# Tuning Lab

> [!IMPORTANT]
> Tuning Lab is under active development

Tuning Lab contains examples for NVIDIA's [NeMo 2.0](https://www.nvidia.com/en-us/ai-data-science/products/nemo/) and Meta's [torchtune](https://github.com/pytorch/torchtune).

# Models

We will use NVIDIA and Meta models including, but not limited to:

- NVIDIA Llama variants, Mistral variants, Megatron distillations, and Minitron
- Meta Llama variants

# User Account Requirements

- [NVIDIA Developer Program](https://developer.nvidia.com/developer-program)
- [NVIDIA NGC](https://catalog.ngc.nvidia.com/) 
- [Hugging Face Hub](https://huggingface.co/) 
- [Weights & Biases](https://wandb.ai/site)

# NeMo Instructions

## System Requirements

- a CUDA compatible OS and device (GPU) with at least 48GB of VRAM (e.g. an L40S).
- CUDA 12.1
- Python 3.10.10
- Pytorch 2.2.1

## Dependencies 

- [Megatron Core](https://github.com/NVIDIA/Megatron-LM)
- [APEX](https://github.com/NVIDIA/apex)
- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)


## On Host (local, no container)

To prepare a development environment, please run the following in terminal:

```sh
bash install_nemo_requirements.sh
```

Doing so will install `tuning_lab` along with `nemo_run`, `megatron_core 0.10.0rc0`, and the `nvidia/apex` PyTorch extension. 

> [!NOTE]
> `megatron_core 0.10.0rc0` is required for compatibility with NeMo 2.0

> [!NOTE]
> NVIDIA Apex is required for RoPE Scaling in NeMo 2.0.
> NVIDIA Apex is built with CUDA and C++ extensions for performance and full functionality.
> please be aware that the build process may take several minutes

## Docker

> [!IMPORTANT]
> running the images requires for the host machine to have access to NVIDIA GPUs

Two Docker images have been created for the quick start tutorials. One for pretraining, and one for finetuning.

To run pretraining, do the following in terminal:

```sh
docker pull jxtngx/nemo-lab:pretrain
docker run --rm --gpus 1 -it jxtngx/nemo-lab:pretrain
python pretrain_nemotron3_4b.py
```

To run finetuning, do the following in terminal:

```sh
docker pull jxtngx/nemo-lab:finetune
docker run --rm --gpus 1 -it jxtngx/nemo-lab:finetune
# WAIT FOR CONTAINER TO START 
huggingface-cli login
# ENTER HF KEY WHEN PROMPTED
python finetune_llama3_8b.py
```

> [!IMPORTANT]
> Finetuning requires a Hugging Face key and access to Llama 3 8B <br/>
> For keys, see: https://huggingface.co/docs/hub/en/security-tokens <br/>
> For Llama 3 8B access, see: https://huggingface.co/meta-llama/Meta-Llama-3-8B

## Quickstart Images and Containers

<table>
    <tr>
        <th style="font-size: small;">Quickstart</th>
        <th style="font-size: small;">Docker</th>
        <th style="font-size: small;">NVIDIA</th>
        <th style="font-size: small;">Lightning</th>
        <th style="font-size: small;">Modal</th>
        <th style="font-size: small;">Hugging Face</th>
    </tr>
    <tr>
        <td>NeMo Pretrain</td>
        <td><a target="_blank" href="https://hub.docker.com/repository/docker/jxtngx/nemo-lab/tags/train/sha256-e056f7988b875676e01a0ba2ae8fa4623ae28f0295a012f809e98158f413c2a2"><img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=fff"/></a></td>
        <td><a target="_blank" href="https://console.brev.dev/launchable/deploy?launchableID=env-2qcfxLVNihSWulFbpOYJmyy8kBB"><img src="https://uohmivykqgnnbiouffke.supabase.co/storage/v1/object/public/landingpage/brevdeploynavy.svg"/></a></td>
        <td><a target="_blank" href="https://lightning.ai/jxtngx/studios/pretrain-nemotron-3-4b-with-nvidia-nemo"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a></td>    
        <td>–</td>
        <td>–</td>
    </tr>
    <tr>
        <td>NeMo Finetune</td>
        <td><a target="_blank" href="https://hub.docker.com/repository/docker/jxtngx/nemo-lab/tags/tune/sha256-29c27b9c41d633f48ed92aec40e33438b84fb3751e764df4655c9b2e697650d7"><img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=fff"/></a></td>
        <td><a target="_blank" href="https://console.brev.dev/launchable/deploy?launchableID=env-2qcfslJNcRDEkHNGfhpW625D2SN"><img src="https://uohmivykqgnnbiouffke.supabase.co/storage/v1/object/public/landingpage/brevdeploynavy.svg"/></a></td>
        <td><a target="_blank" href="https://lightning.ai/jxtngx/studios/finetune-llama-3-8b-with-nvidia-nemo"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/></a></td>
        <td>–</td>
        <td>–</td>
    </tr>
</table>

> [!IMPORTANT]
> regarding the NVIDIA Launchable, use the following command in terminal to run the: <br/>
> tuning: `python /workspace/finetune_llama3_8b.py` <br/>
> training: `python /workspace/pretrain_nemotron3_4b.py` <br/>
> to avoid data storage costs, be certain to delete the instance once the demo is complete 

## References

- [NeMo documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)
- [NeMo tutorials](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/tutorials.html)
- [NeMo Guardrails documentation](https://docs.nvidia.com/nemo/guardrails/index.html)
- [Deploy on a SLURM cluster](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html#execute-on-a-slurm-cluster)
- [Mixed Precision Training](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/mixed_precision.html)
- [CPU Offloading](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/optimizations/cpu_offloading.html)
- [Communication Overlap](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/optimizations/communication_overlap.html)

# Torchtune Instructions

## System Requirements

- a torch compatible development environment and appropriate hardware for a target model

## Quickstart Images and Containers

<table>
    <tr>
        <th style="font-size: small;">Quickstart</th>
        <th style="font-size: small;">Docker</th>
        <th style="font-size: small;">NVIDIA</th>
        <th style="font-size: small;">Lightning</th>
        <th style="font-size: small;">Modal</th>
        <th style="font-size: small;">Hugging Face</th>
    </tr>
    <tr>
        <td>NeMo Pretrain</td>
        <td>–</td>
        <td>–</td>   
        <td>–</td>
        <td>–</td>
        <td>–</td>
    </tr>
</table>