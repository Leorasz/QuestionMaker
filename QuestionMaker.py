from datasets import load_dataset

def download_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset

# Use the function to download the 'squad' dataset
capybara_dataset = download_dataset('LDJnr/Capybara')

"""
TODO:
-Find good LLM
-Feed capybara stuff into them
-Make good prompt structure
-Handle results
-Ensemble multiple LLMs?
-Validate with ANN?
-Validate with other LLMs?
"""

import torch
import transformers

model_id = 'meta-llama/Llama-2-7b-chat-hf'

if torch.cuda.is_available():
    device = f'cuda:{torch.cuda.current_device}'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
# begin initializing HF items, need auth token for these
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    device_map='auto',
    token=hf_auth
)
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    token=hf_auth
)

get_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.01,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

def generate_text(prompt):
    res = get_text(prompt)
    return res[0]["generated_text"]

print(generate_text("What is the answer to life, the universe, and everything"))
