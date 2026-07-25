#!/usr/bin/env python
"""
Modal App for SWE-Qwen Training and Inference

This module defines the Modal application with GPU-enabled functions
for training, fine-tuning, and serving the SWE-Qwen model.

Primary model: Qwen/Qwen3-30B-A3B (30B MoE, 3B active params)
Fallback model: Qwen/Qwen3-14B (14B dense)
"""

import modal

# Modal app configuration
app = modal.App("swe-qwen")

# Base image with all dependencies
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "wget",
        "curl",
        "build-essential",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgl1-mesa-glx",
    )
    .pip_install(
        "torch==2.11.0",
        "torchvision==0.16.0",
        "torchaudio==2.11.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "transformers>=5.14.0",
        "accelerate>=1.14.0",
        "peft>=0.19.0",
        "bitsandbytes>=0.49.0",
        "trl>=1.9.0",
        "datasets>=5.0.0",
        "wandb>=0.28.0",
        "vllm>=0.26.0",
        "flash-attn>=2.7.0",
        "deepspeed>=0.16.0",
        "sentencepiece>=0.2.0",
        "protobuf>=5.28.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
    )
)

# GPU image for training (larger, with more ML libs)
gpu_image = base_image.pip_install(
    "xformers>=0.0.27",
    index_url="https://download.pytorch.org/whl/cu121",
).pip_install(
    "auto-gptq>=0.7.1",
    "optimum>=1.21.0",
)

# CPU image for lightweight tasks
cpu_image = base_image


@app.function(
    image=cpu_image,
    secrets=[
        modal.Secret.from_name("modal-api-token"),
        modal.Secret.from_name("wandb-api-key"),
        modal.Secret.from_name("github-token"),
    ],
    timeout=300,
)
def hello_modal():
    """Simple test function to verify Modal setup."""
    import os

    return {
        "status": "ok",
        "message": "Modal is configured correctly!",
        "modal_token_set": bool(os.environ.get("MODAL_TOKEN")),
        "wandb_key_set": bool(os.environ.get("WANDB_API_KEY")),
    }


@app.function(
    image=gpu_image,
    gpu="A100:1",  # QLoRA 4-bit fits A100 40GB; upgrade to H100:1 if needed
    secrets=[
        modal.Secret.from_name("modal-api-token"),
        modal.Secret.from_name("wandb-api-key"),
        modal.Secret.from_name("github-token"),
    ],
    volumes={
        "/data": modal.Volume.from_name("swe-qwen-datasets", create_if_missing=True),
        "/models": modal.Volume.from_name("swe-qwen-models", create_if_missing=True),
    },
    timeout=86400,  # 24 hours max
    retries=modal.Retries(
        max_retries=2,
        backoff_coefficient=2.0,
        initial_delay=60.0,
    ),
)
def train_swe_qwen(  # noqa: PLR0913
    model_name: str = "Qwen/Qwen3-30B-A3B",  # fallback: Qwen/Qwen3-14B
    dataset_path: str = "/data/train",
    output_dir: str = "/models/swe-qwen-finetuned",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    use_4bit: bool = True,
    use_flash_attn: bool = True,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 4096,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    wandb_project: str = "swe-qwen",
    wandb_run_name: str | None = None,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
):
    """
    Fine-tune SWE-Qwen model using LoRA on Modal GPUs.

    Args:
        model_name: Base model to fine-tune
        dataset_path: Path to training dataset
        output_dir: Output directory for checkpoints
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        use_4bit: Use 4-bit quantization
        use_flash_attn: Use Flash Attention 2
        gradient_accumulation_steps: Gradient accumulation steps
        max_seq_length: Maximum sequence length
        warmup_ratio: Warmup ratio
        weight_decay: Weight decay
        logging_steps: Logging interval
        save_steps: Save interval
        eval_steps: Evaluation interval
        wandb_project: W&B project name
        wandb_run_name: W&B run name
        push_to_hub: Push to Hugging Face Hub
        hub_model_id: Hub model ID
    """
    import os
    from datetime import datetime

    import torch
    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from trl import SFTTrainer

    import wandb

    # Set up W&B
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    run_name = wandb_run_name or f"swe-qwen-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project=wandb_project, name=run_name, config=locals())

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load dataset
    dataset = load_from_disk(dataset_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation", dataset.get("test"))

    # Quantization config
    if use_4bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attn else "eager",
    )

    # Prepare for k-bit training
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        bf16=True,
        tf32=True,
        report_to="wandb",
        run_name=run_name,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        formatting_func=lambda x: x["text"],
        max_seq_length=max_seq_length,
        packing=True,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Push to hub if requested
    if push_to_hub and hub_model_id:
        model.push_to_hub(hub_model_id, use_temp_dir=True)
        tokenizer.push_to_hub(hub_model_id, use_temp_dir=True)

    # Log artifacts to W&B
    artifact = wandb.Artifact(f"model-{run_name}", type="model")
    artifact.add_dir(output_dir)
    wandb.log_artifact(artifact)

    wandb.finish()

    return {
        "status": "completed",
        "output_dir": output_dir,
        "run_name": run_name,
    }


@app.function(
    image=gpu_image,
    gpu="H100:1",
    secrets=[
        modal.Secret.from_name("wandb-api-key"),
    ],
    volumes={
        "/models": modal.Volume.from_name("swe-qwen-models", create_if_missing=True),
    },
    timeout=3600,
    concurrency_limit=4,
    scaledown_window=300,
)
async def serve_swe_qwen(  # noqa: PLR0913
    model_path: str = "/models/swe-qwen-finetuned",
    host: str = "0.0.0.0",
    port: int = 8000,
    max_model_len: int = 4096,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
):
    """
    Serve SWE-Qwen model using vLLM for fast inference.

    Args:
        model_path: Path to the fine-tuned model
        host: Host to bind to
        port: Port to listen on
        max_model_len: Maximum model context length
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization fraction
    """
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    # Initialize vLLM engine
    engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Create FastAPI app
    app = FastAPI(title="SWE-Qwen API", version="1.0.0")

    @app.get("/health")
    async def health():
        return {"status": "healthy", "model": model_path}

    @app.post("/v1/completions")
    async def completions(request: dict):
        from vllm.entrypoints.openai.protocol import CompletionRequest

        req = CompletionRequest(**request)
        generator = engine.generate(
            prompt=req.prompt,
            sampling_params=req.sampling_params,
            request_id=req.request_id,
        )
        return StreamingResponse(generator, media_type="application/json")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict):
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest

        req = ChatCompletionRequest(**request)
        generator = engine.chat(
            messages=req.messages,
            sampling_params=req.sampling_params,
            request_id=req.request_id,
        )
        return StreamingResponse(generator, media_type="application/json")

    # Run server
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

    return {"status": "serving", "model": model_path}


@app.local_entrypoint()
def main():
    """Local entrypoint for testing."""
    print("Testing Modal setup...")
    result = hello_modal.remote()
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
