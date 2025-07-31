#imports
import torch 
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training)
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initialize dataset
DATASET_NAME = "ChrisHayduk/Llama-2-SQL-Dataset" 
#load the dataset
dataset = load_dataset(DATASET_NAME)

full_training_dataset = dataset["train"]
shuffled = full_training_dataset.shuffle() #calling .shuffle() shuffles the dataset randomly
training_dataset = shuffled.select(range(1000))  # Select first 1000 samples for fine tuning

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16") 

MODEL_NAME = "NousResearch/Llama-2-7b-hf"  # Specify the model name

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config, #use the above quantization config
    device_map="auto" ) 

model.config.use_cache = True

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code = True) 

tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
tokenizer.padding_side = "right"  # Set padding side to right

#preparing the prompts in the format for fine-tuning the model
def construct_datapoint(x): 
    combined = x['input'] + x['output'] 
    return tokenizer(combined, padding = True) 
training_dataset = training_dataset.map(construct_datapoint) #passing in the function to every element in the dataset to map the input and output to the tokenizer  

#defining our LORA (low rank adaptation) config
peft_config = LoraConfig(
    r=16, #higher r means more fine tuned parameters but more memory usage
    lora_alpha=32, 
    #which layers to apply LORA to below, q for query, k for key, v for value, o for output 
    target_modules=['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj' , 'up_proj'],
    lora_dropout=0.05, # Dropout prevents overfitting, every iteration it will randomly turn off some nodes to 0, overfititng is when the model memorizes random noise in the training data
    task_type = "CAUSAL_LM" 
)

model = prepare_model_for_kbit_training(model)  # Prepare the model for k-bit training, storing weights in 4 bits
model = get_peft_model(model, peft_config)  # Apply the PEFT model configuration, all layers except attention laters are frozen

generation_configuration = model.generation_config
generation_configuration.pad_token_id = tokenizer.eos_token_id
generation_configuration.eos_token_id = tokenizer.eos_token_id
generation_configuration.max_new_tokens = 256
generation_configuration.temperature = 0.7
generation_configuration.top_p = 0.9
generation_configuration.do_sample = True

def generaate(prompt) : 
    generation_configuration.max_new_tokens = 20 

    encoded = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
    with torch.inference_mode(): 
        out = model.generate(input_ids=encoded, generation_config=generation_configuration, repetition_penalty = 2.0)
    string_decoded = tokenizer.decode(out[0], clean_up_tokenization_spaces=True)
    print(string_decoded) 


train_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=1,  # Batch size per device during training
    gradient_accumulation_steps=4,  # Number of updates steps to accumulate before performing a backward/update pass
    num_train_epochs=1,  # Total number of training epochs
    learning_rate=2e-4,  # Initial learning rate
    fp16 = True,
    optim="paged_adamw_8bit",
    lr_scheduler_type = "cosine",  # Learning rate scheduler type
    warmup_ratio = 0.05,
    output_dir= "fine-tuning") 

#initialize the trainer

trainer = transformers.Trainer( 
    model = model,
    train_dataset = training_dataset,
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), 
    args = train_arguments 
)

model.config.use_cache = False  # Disable cache during training
