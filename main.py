import copy # allows to create independent copy of an existing object
import random
import re # functionalities for regular expressions
import time # functionalities for time manipulation / measure elapsed time
from typing import Any, Literal, TypedDict # fucntionalities for working with data types
import re

import torch # core library for tensor computation
import wandb # weights & biases integration, visualizing & tracking learning metrics, model performance
import transformers # Hugging Face's NLP models + functionalities for training, evaluation, deployment of these models
from datasets import Dataset, DatasetDict # lib developed bu HF: provides access to datasets of ML
from jaxtyping import Int # module providing type annotations for JAX: lib for parallel and distributed numerical computation
from torch import FloatTensor, Tensor # represent floating-point tensors in PyTorch
from torch.optim import Adam # optimization algorithm for deep neural network training
from tqdm.auto import tqdm
from transformers import (
    GenerationConfig,
    GPTNeoConfig,
    GPTNeoForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
) # output of a conditional language model with cross attentions
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import evaluate

Path("./data/").mkdir(parents=True, exist_ok=True)

# Load the TinyStories model
tinystories_model = transformers.AutoModelForCausalLM.from_pretrained(
    "roneneldan/TinyStories-1M", revision="8cd14d5", cache_dir="./data/"
)

# Create a random version of this model (by re-calling the i)
random_init_model = transformers.AutoModelForCausalLM.from_pretrained(
    "roneneldan/TinyStories-1M", revision="8cd14d5", cache_dir="./data/"
)
random_init_model.apply(random_init_model._init_weights)  # noqa: SLF001

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "roneneldan/TinyStories-1M",
    revision="8cd14d5",
    cache_dir="./data/",
    padding_side="left",  # Left padding so generate works
)
tokenizer.pad_token = tokenizer.eos_token

# ----------------------------------
# TASK 1: CREATE A SYNTHETIC DATASET
# ----------------------------------

def count_in_unary(start, end):
    completion = ""

    for i in range(len(start), len(end) + 1):
        completion += " " + "1" * i

    pad_id = tokenizer.pad_token_id
    completion += tokenizer.decode(pad_id)

    return completion

def generate_unary_number(stop):
    prompts = []
    completions = []
    
    for i in range(1, stop):
        for j in range(i + 1, stop):
            if (len(prompts) >= stop):
                break
            if stop == 70 and j % 10 not in [1, 3, 7]:
                prompts.append("Please count up in unary, starting at " + "1" * i + " and stopping at " + "1" * j + ":")
                completions.append(count_in_unary(i * "1", j * "1"))
            elif stop == 30 and j % 10 in [1, 3, 7]:
                prompts.append("Please count up in unary, starting at " + "1" * i + " and stopping at " + "1" * j + ":")
                completions.append(count_in_unary(i * "1", j * "1"))
    
    return Dataset.from_dict({ "prompt": prompts, "completion": completions })

def create_dataset() -> DatasetDict:
    train_dataset = generate_unary_number(70)
    test_dataset = generate_unary_number(30)

    return DatasetDict({"test": test_dataset, "train": train_dataset})

# -----------------------
# TASK 2: EVALUATE MODELS
# -----------------------

# prediction generation: token by token
# contextualized predictions
def evaluate_model_ar(
    model: PreTrainedModel,
    dataset: Dataset,
    pre_trained_tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 8,
    context_window_size: int = 1536, 
) -> float:
    
    correct_answers = 0
    total_answers = 0
    start_time = time.time()
    model_config = model.config
    model_config.pad_token_id = pre_trained_tokenizer.pad_token_id
    dataset_batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

    for batch in tqdm(dataset_batches, desc="Evaluating model - auto-regression", total=len(dataset_batches)):
        prompts = batch["prompt"]
        completions = batch["completion"]

        inputs = pre_trained_tokenizer(prompts, truncation=True, padding=True, max_length=context_window_size, return_tensors='pt')
        
        with torch.no_grad():
            predictions = model.generate(**inputs, max_length=context_window_size, do_sample=False, pad_token_id=model_config.pad_token_id)

            predicted_completions = [pre_trained_tokenizer.decode(pred, skip_special_tokens=False) for pred in predictions]

        for predicted_completion, expected_completion in zip(predicted_completions, completions):
            if predicted_completion == expected_completion:
                correct_answers += 1

            total_answers += len(batch)

    print('execution time: ', time.time() - start_time)
    return correct_answers / total_answers

# predictions for an entire sequence using batch
# parallelized processes
def evaluate_model(
    model: PreTrainedModel,
    dataset: Dataset,
    pre_trained_tokenizer: PreTrainedTokenizerBase, # preprocess textual data before passing it to the model
    batch_size: int = 8, # nb of dataset used at the same time when evaluating
    context_window_size: int = 1536, # nb of context's tokens used when evaluating
) -> float:
    
    # model.config.temperature = 0.0
    correct_answers = 0
    total_answers = 0
    start_time = time.time()
    dataset_batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

    for batch in tqdm(dataset_batches, desc="Evaluating model - non-regression", total=len(dataset_batches), colour="green"):
        prompts = batch["prompt"]
        completions = batch["completion"]

        inputs = pre_trained_tokenizer(prompts, truncation=True, padding=True, max_length=context_window_size, return_tensors='pt')
        # desactivate gradient calculation during inference + optimization
        with torch.no_grad():
            # get model's predictions for every prompt (= inference stage)
            outputs = model.forward(**inputs)
        
            # get logits: scores associated with each token / model's confidence in its predictions for each token
            # get most probable prediction
            predictions = torch.argmax(outputs.logits, dim=-1)

            # decoding the most probable predictions for the batch
            predicted_completions = [pre_trained_tokenizer.decode(pred, skip_special_tokens=False) for pred in predictions]

            for predicted_completion, expected_completion in zip(predicted_completions, completions):
                # print('pred: ', predicted_completion)
                if predicted_completion == expected_completion:
                    correct_answers += 1
       
                total_answers += 1

    print('execution time: ', time.time() - start_time)
    return correct_answers / total_answers

# ----------------------------
# TASK 3: TEST YOUR EVALUATOR
# ----------------------------

class DummyModel(GPTNeoForCausalLM):
    """Dummy model that can do unary counting completion."""

    def __init__(
        self,
        config: GPTNeoConfig = transformers.AutoConfig.from_pretrained("roneneldan/TinyStories-1M"),
        unary_accuracy: float = 0.5,
    ) -> None:
        """Initialize the model."""
        super().__init__(config)
        self.unary_accuracy = unary_accuracy
        self.model = GPTNeoForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M",
            revision="8cd14d5",
            cache_dir="./dummy_data/",
            padding_side="left",
        )
        self.tokenizer.pad_token = tokenizer.eos_token

    def forward(
        self,
        input_ids: Tensor | None = None,
        past_key_values: tuple[FloatTensor] | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        head_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor] | CausalLMOutputWithCrossAttentions:
        """Forward pass."""

        def get_correct_completion():
            prompt_text = self.tokenizer.decode(input_ids[item])
            start, end = [s.strip() for s in re.findall(r'\b\d+\b', prompt_text)]
            completion = count_in_unary(start, end)
            
            return self.tokenizer.encode(completion, add_special_tokens=False)

        def get_wrong_completion():
            completion = "wrong token<|endoftext|>"
            
            return self.tokenizer.encode(completion, add_special_tokens=False)
        
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        buff = []
        max_sequence_length = 0

        for item in range(batch_size):
            if random.uniform(0., 1.) < self.unary_accuracy:
                completion_ids = get_correct_completion()
            else:
                completion_ids = get_wrong_completion()

            sequence_length = len(completion_ids)
            single_logit = torch.zeros((sequence_length, self.config.vocab_size))

            for i in range(sequence_length):
                single_logit[i, completion_ids[i]] = 1.0
            
            buff.append(single_logit)
            max_sequence_length = max(max_sequence_length, sequence_length)

        logits = torch.zeros((batch_size, max_sequence_length, self.config.vocab_size))

        for item in range(batch_size):
            sequence_length = buff[item].size(0)
            logits[item, :sequence_length, :] = buff[item]

        output = CausalLMOutputWithCrossAttentions(
            logits=logits,
            past_key_values=past_key_values,
        )

        return output

# --------------------------
# TASK 4: TRANSFER LEARNING
# --------------------------

def train_model(
    model: PreTrainedModel,
    text_dataset: DatasetDict,
    learning_rate: float = 0.001,
    num_epochs: int = 3, 
    batch_size: int = 1,
    pre_trained_tokenizer: PreTrainedTokenizerBase = tokenizer,
) -> None:
    
    """Train the model with the given dataset and learning rate.

    Returns:
        The accuracy of the model on the validation set.
    """

    # prbl -> model return prompt & not completion after training
    def preprocess_data(dataset):
        inputs = pre_trained_tokenizer(dataset["prompt"], padding=True, return_tensors="pt")
        outputs = pre_trained_tokenizer(dataset["completion"], padding=True, return_tensors="pt").input_ids
        # pair prompts w/ tokens of completions
        # used to calculate loss during training
        inputs["labels"] = outputs

        return inputs

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)
        # get most probable predictions
        predictions = torch.argmax(logits, dim=-1)
        # flatten predictions & labels
        predictions = predictions.view(-1)
        labels = labels.view(-1)

        return metric.compute(predictions=predictions, references=labels)
    
    metric = evaluate.load("accuracy")

    train_dataset = text_dataset["train"]
    eval_dataset = text_dataset["test"]
    train_dataset = train_dataset.map(preprocess_data, batched=True)
    eval_dataset = eval_dataset.map(preprocess_data, batched=True)

    # update model's params during training
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # enable to create batchs
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=pre_trained_tokenizer, mlm=False)

    training_args = transformers.TrainingArguments(
        output_dir="my_awesome_model",
        evaluation_strategy="epoch",
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )

    trainer.train()
    eval_results = trainer.evaluate(eval_dataset)

    return eval_results["eval_accuracy"]

# Task 1
def test_implementation(dataset):
    example_train_prompt = "Please count up in unary, starting at 1 and stopping at 11:"
    example_train_completion = " 1 11<|endoftext|>"

    train_match = [i for i in dataset["train"] if i["prompt"] == example_train_prompt]  # type: ignore
    assert len(train_match) == 1
    assert train_match[0]["completion"] == example_train_completion  # type: ignore

    example_test_prompt = "Please count up in unary, starting at 1 and stopping at 111:"
    unary_test_completion = " 1 11 111<|endoftext|>"
    test_match = [i for i in dataset["test"] if i["prompt"] == example_test_prompt]  # type: ignore
    assert len(test_match) == 1
    assert test_match[0]["completion"] == unary_test_completion  # type: ignore

# Task 2
def evaluate_task(dataset):
    tinystories_model_acc = evaluate_model(tinystories_model, dataset["test"], tokenizer)
    evaluate_model_ar(tinystories_model, dataset["test"], tokenizer)

    random_init_model_acc = evaluate_model(random_init_model, dataset["test"], tokenizer)
    evaluate_model_ar(random_init_model, dataset["test"], tokenizer)

    print(f"tinystories_model accuracy: {tinystories_model_acc}")
    print(f"random_init_model accuracy: {random_init_model_acc}")

# Task 3
def test_evaluator(dataset):
    dataset_single_item = dataset["test"].select([2])  # We could have picked any item here

    test_0_accuracy = evaluate_model(DummyModel(unary_accuracy=0), dataset_single_item, tokenizer, batch_size=1)
    assert test_0_accuracy == 0.0, "The accuracy should be 0."

    test_1_accuracy = evaluate_model(DummyModel(unary_accuracy=1), dataset_single_item, tokenizer, batch_size=1)
    assert test_1_accuracy == 1.0, "The accuracy should be 1."

    # # Test a partially accurate model
    tested_probability = 0.3
    dummy_model = DummyModel(unary_accuracy=tested_probability)
    model_accuracy = evaluate_model(dummy_model, dataset["test"], tokenizer, batch_size=2)
    print(f" The accuracy of your DummyModel is {model_accuracy}, the expected accuracy should be c. {tested_probability}")
    assert (
        tested_probability - 0.2 < model_accuracy < tested_probability + 0.2
    ), "The model should be close to the expected accuracy"

def transfer_learning(dataset):
    # Create an appropriate dataset
    full_text_dataset = create_dataset()

    t1 = time.time()

    train_model(random_init_model, full_text_dataset)
    train_model(tinystories_model, full_text_dataset)

    t2 = time.time()

    # should take < 5 minutes
    print(f"Model training time: {(int((t2-t1)/60))} minutes")

    random_init_model_accuracy = evaluate_model(random_init_model, dataset["test"], tokenizer)
    tinystories_model_accuracy = evaluate_model(tinystories_model, dataset["test"], tokenizer)

    print(f"random_init_model accuracy: {random_init_model_accuracy}")
    print(f"tinystories_model accuracy: {tinystories_model_accuracy}")

dataset = create_dataset()

# test_implementation(dataset) # Task 1
# evaluate_task(dataset) # Task 2
# test_evaluator(dataset) # Task 3
transfer_learning(dataset) # Task 4
