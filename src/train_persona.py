"""
Script for fine-tuning persona models using Claude API.
Based on the structure of paul_graham_chat_finetuning.py but adapted for Anthropic Claude API.
"""

import os
import time
import json
from typing import Any, Optional, List, Dict
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
from anthropic import Anthropic
from characters.base import CharacterBase
from characters.paul_graham import PaulGraham

# Load environment variables from .env file
# Try loading from project root first, then from current directory
env_paths = [
    Path(__file__).parent.parent / ".env",  # Project root
    Path(__file__).parent / ".env",  # src directory
    Path(".env"),  # Current directory
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break


@dataclass
class TrainingArguments:
    """
    Training arguments for Claude API fine-tuning.
    Similar structure to HuggingFace's TrainingArguments.
    """
    output_dir: str = "./claude-fine-tuned"
    evaluation_strategy: str = "epoch"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    logging_dir: str = "./logs"
    logging_steps: int = 10
    
    def to_hyperparameters(self) -> Dict[str, Any]:
        """
        Convert TrainingArguments to Claude API hyperparameters format.
        
        Returns:
            Dictionary of hyperparameters for Claude API
        """
        # Map TrainingArguments to Claude API parameters
        # Note: Actual parameter names depend on Claude API documentation
        hyperparameters = {
            "num_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.per_device_train_batch_size,
            "weight_decay": self.weight_decay,
            # Add other mappings as needed based on Claude API support
        }
        return hyperparameters


def prepare_dataset(
    hf_dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    output_file: Optional[str] = None
) -> str:
    """
    Prepare dataset for Claude API fine-tuning.
    
    Claude API expects training data in JSONL format where each line is:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    
    Args:
        hf_dataset_name: HuggingFace dataset name (e.g., "pookie3000/pg_chat")
        dataset_path: Local path to dataset file
        output_file: Path to save the formatted JSONL file
        
    Returns:
        Path to the prepared JSONL file
    """
    if output_file is None:
        output_file = "training_data.jsonl"
    
    # Load dataset
    if hf_dataset_name:
        dataset = load_dataset(hf_dataset_name, split="train")
    elif dataset_path:
        # Assume it's a JSON or JSONL file
        if dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r') as f:
                data = [json.loads(line) for line in f]
            dataset = Dataset.from_list(data)
        else:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            dataset = Dataset.from_list(data if isinstance(data, list) else [data])
    else:
        raise ValueError("Either hf_dataset_name or dataset_path must be provided")
    
    # Format for Claude API
    formatted_data = []
    for example in dataset:
        # Handle different dataset formats
        if "conversations" in example:
            # Format: {"conversations": [{"role": "user", "content": "..."}, ...]}
            messages = []
            for msg in example["conversations"]:
                role = msg.get("role", "user")
                content = msg.get("content", msg.get("text", ""))
                messages.append({"role": role, "content": content})
            formatted_data.append({"messages": messages})
        elif "messages" in example:
            # Already in correct format
            formatted_data.append({"messages": example["messages"]})
        elif "text" in example:
            # Single text field - might need to split or format differently
            # This is a fallback - you may need to adjust based on your data
            formatted_data.append({
                "messages": [
                    {"role": "user", "content": "Continue the conversation"},
                    {"role": "assistant", "content": example["text"]}
                ]
            })
    
    # Write to JSONL file
    with open(output_file, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Prepared {len(formatted_data)} examples and saved to {output_file}")
    return output_file


def upload_training_file(client: Anthropic, file_path: str) -> str:
    """
    Upload training file to Anthropic.
    
    Args:
        client: Anthropic API client
        file_path: Path to the JSONL training file
        
    Returns:
        File ID for the uploaded file
    """
    print(f"Uploading training file: {file_path}")
    
    with open(file_path, 'rb') as f:
        file = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    
    print(f"File uploaded successfully. File ID: {file.id}")
    return file.id


def create_fine_tuning_job(
    client: Anthropic,
    training_file_id: str,
    model: str = "claude-3-5-sonnet-20241022",
    training_args: Optional[TrainingArguments] = None,
    hyperparameters: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a fine-tuning job using Claude API.
    
    Args:
        client: Anthropic API client
        training_file_id: ID of the uploaded training file
        model: Base model to fine-tune (e.g., "claude-3-5-sonnet-20241022")
        training_args: TrainingArguments object (takes precedence over hyperparameters)
        hyperparameters: Optional hyperparameters for training (used if training_args not provided)
        
    Returns:
        Fine-tuning job ID
    """
    print(f"Creating fine-tuning job for model: {model}")
    
    # Use TrainingArguments if provided, otherwise use hyperparameters dict
    if training_args is not None:
        hyperparameters = training_args.to_hyperparameters()
        print(f"Using TrainingArguments: {training_args}")
    elif hyperparameters is None:
        hyperparameters = {}
    
    # Create fine-tuning job
    # Note: The exact API may vary - check Anthropic documentation for latest API
    job = client.fine_tuning.jobs.create(
        model=model,
        training_file_id=training_file_id,
        hyperparameters=hyperparameters
    )
    
    print(f"Fine-tuning job created. Job ID: {job.id}")
    return job.id


def wait_for_job_completion(
    client: Anthropic,
    job_id: str,
    check_interval: int = 60
) -> Dict[str, Any]:
    """
    Wait for fine-tuning job to complete and return the result.
    
    Args:
        client: Anthropic API client
        job_id: Fine-tuning job ID
        check_interval: Seconds between status checks
        
    Returns:
        Job result dictionary with model ID and status
    """
    print(f"Waiting for job {job_id} to complete...")
    
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        
        print(f"Job status: {status}")
        
        if status == "succeeded":
            print(f"Fine-tuning completed successfully!")
            print(f"Fine-tuned model ID: {job.fine_tuned_model_id}")
            return {
                "status": "succeeded",
                "model_id": job.fine_tuned_model_id,
                "job_id": job_id
            }
        elif status == "failed":
            error = getattr(job, "error", "Unknown error")
            print(f"Fine-tuning failed: {error}")
            return {
                "status": "failed",
                "error": error,
                "job_id": job_id
            }
        elif status in ["validating", "queued", "running"]:
            print(f"Job is {status}. Waiting {check_interval} seconds...")
            time.sleep(check_interval)
        else:
            print(f"Unknown status: {status}")
            time.sleep(check_interval)


def train(
    character: CharacterBase,
    hf_dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    base_model: str = "claude-3-5-sonnet-20241022",
    training_args: Optional[TrainingArguments] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Train a persona model using Claude API fine-tuning.
    
    Args:
        character: Character instance (e.g., PaulGraham) with dataset_path
        hf_dataset_name: HuggingFace dataset name (optional, uses character.dataset_path if not provided)
        dataset_path: Local dataset path (optional, uses character.dataset_path if not provided)
        base_model: Base Claude model to fine-tune
        training_args: TrainingArguments object (preferred over hyperparameters)
        hyperparameters: Training hyperparameters dict (used if training_args not provided)
        api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
        
    Returns:
        Fine-tuned model ID/checkpoint path
    """
    # Initialize Anthropic client
    # Load from .env file if not provided
    if not api_key:
        load_dotenv()  # Ensure .env is loaded
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY must be set in .env file or provided as parameter. "
            "Create a .env file in the project root with: ANTHROPIC_API_KEY=your_key_here"
        )
    
    client = Anthropic(api_key=api_key)
    
    # Determine dataset source
    if not hf_dataset_name and not dataset_path:
        dataset_path = character.dataset_path
        if not dataset_path:
            raise ValueError("Dataset path must be provided via character.dataset_path or parameters")
    
    # Prepare dataset
    training_file = prepare_dataset(
        hf_dataset_name=hf_dataset_name,
        dataset_path=dataset_path,
        output_file="training_data.jsonl"
    )
    
    # Upload training file
    file_id = upload_training_file(client, training_file)
    
    # Create fine-tuning job
    job_id = create_fine_tuning_job(
        client=client,
        training_file_id=file_id,
        model=base_model,
        training_args=training_args,
        hyperparameters=hyperparameters
    )
    
    # Wait for completion
    result = wait_for_job_completion(client, job_id)
    
    if result["status"] == "succeeded":
        model_id = result["model_id"]
        # Update character with checkpoint path
        character.persona_llm_ckpt_path = model_id
        print(f"Fine-tuned model ID saved to character: {model_id}")
        return model_id
    else:
        raise RuntimeError(f"Fine-tuning failed: {result.get('error', 'Unknown error')}")


def main():
    """
    Main entry point for training persona models.
    """
    # PARAMS - Similar structure to paul_graham_chat_finetuning.py
    hf_dataset_name = "pookie3000/pg_chat"  # Optional: use HuggingFace dataset
    # dataset_path = None  # Or use local dataset path
    
    base_model = "claude-3-5-sonnet-20241022"  # Base Claude model
    
    # Training arguments - similar to HuggingFace TrainingArguments
    training_args = TrainingArguments(
        output_dir="./claude-fine-tuned",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10
    )
    
    # Create character instance
    character = PaulGraham(
        dataset_path=None,  # Will use hf_dataset_name instead
        persona_llm_ckpt_path=None  # Will be set after training
    )
    
    # TRAINING
    try:
        model_id = train(
            character=character,
            hf_dataset_name=hf_dataset_name,
            base_model=base_model,
            training_args=training_args,
        )
        
        print(f"\n{'='*50}")
        print(f"Training completed successfully!")
        print(f"Fine-tuned model ID: {model_id}")
        print(f"Character checkpoint path: {character.persona_llm_ckpt_path}")
        print(f"{'='*50}\n")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()

