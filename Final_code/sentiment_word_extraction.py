import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from collections import defaultdict
from peft import PeftModel
from tqdm import tqdm

# ----------------------------- #
#        Configuration          #
# ----------------------------- #

# Path to the CSV data
DATA_PATH = '/mnt/petrelfs/chenlingjie/google-research/goemotions/data/test.tsv'
use_lora = False
# Paths to the tokenizer and model
TOKENIZER_PATH = "/mnt/hwfile/llm-safety/models/huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct"
ORIGINAL_MODEL_PATH = "/mnt/hwfile/llm-safety/models/huggingface/meta-llama/Meta-Llama-3.1-70B-Instruct"
LORA_MODEL_PATH = "/mnt/petrelfs/chenlingjie/google-research/models/meta-llama/Meta-Llama-3.1-8B-Instruct-sst-1-epoch/checkpoint-2714"

# Output path for predictions
OUTPUT_PATH = 'sentiment_words_dict.json'

# Batch size for processing
BATCH_SIZE = 128

# Define sentiment classes as per your dataset
SENTIMENT_CLASSES = ['admiration',
 'amusement',
 'anger',
 'annoyance',
 'approval',
 'caring',
 'confusion',
 'curiosity',
 'desire',
 'disappointment',
 'disapproval',
 'disgust',
 'embarrassment',
 'excitement',
 'fear',
 'gratitude',
 'grief',
 'joy',
 'love',
 'nervousness',
 'optimism',
 'pride',
 'realization',
 'relief',
 'remorse',
 'sadness',
 'surprise',
 'neutral']

# ----------------------------- #
#        Load the Data          #
# ----------------------------- #

print("Loading data...")
csv_data = pd.read_csv(DATA_PATH, sep='\t',header=None)
csv_data.columns = ['text', 'sentiment_index', 'author']
print(f"Total samples: {len(csv_data)}")

# ----------------------------- #
#    Load Tokenizer & Model     #
# ----------------------------- #

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token
if use_lora == False:
    model = AutoModelForCausalLM.from_pretrained(ORIGINAL_MODEL_PATH)
else:
    base_model = AutoModelForCausalLM.from_pretrained(ORIGINAL_MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH, device_map="auto")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()  # Set model to evaluation mode
print(f"Model loaded on {model.device}")

# ----------------------------- #
#        Define Prompt          #
# ----------------------------- #

def create_prompt(text):
    """
    Constructs a prompt to instruct the model for sentiment classification.
    """
    prompt = (
        "Please extract the sentiment word from the following text.\n"
        "Example:\n"
        "1. Text: That game hurt. Output: hurt.\n"
        "2. Man I love reddit. Output: love\n"
        f"Text:{text}\n"
        "You can should output the sentiment words seperated by ','\n Output:"
    )
    return prompt

# ----------------------------- #
#        Inference Loop         #
# ----------------------------- #

print("Starting sentiment classification...")
sentiment_words_dict = defaultdict(list)
num_batches = (len(csv_data) + BATCH_SIZE - 1) // BATCH_SIZE
false_prediction_count = 0
duplex_prediction = 0

with tqdm(range(0, len(csv_data), BATCH_SIZE), total=num_batches, ncols=100) as pbar:
    for i in pbar:
        batch_texts = csv_data['text'][i:i+BATCH_SIZE].tolist()
        batch_sentiment_index = csv_data['sentiment_index'][i:i+BATCH_SIZE].tolist()
        prompts = [create_prompt(text) for text in batch_texts]
        inputs = tokenizer(prompts, return_tensors='pt', padding=True)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=10,        # Adjust based on expected output length
                temperature=0.0,           # Deterministic output
                top_p=0.95,                # Nucleus sampling
                do_sample=False,           # Disable sampling for consistency
                num_return_sequences=1
            )
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for gen_text, sentiment_index_sting in zip(generated_texts, batch_sentiment_index):
            try:
                # breakpoint()
                sentiment_word_string = gen_text.split('Output:')[-1].strip().rstrip('.')
                sentiment_word_list = sentiment_word_string.split(' ')
                sentiment_index_list = sentiment_index_sting.split(',')
                for sentiment_index in sentiment_index_list:
                    sentiment_words_dict[sentiment_index].extend(sentiment_word_list)
            except Exception as e:
                print(e)
        
        current_batch = i + BATCH_SIZE
        # pbar.set_postfix({
        #     "None Prediction": false_prediction_count / max(current_batch, 1), 
        #     "Duplex Prediction": duplex_prediction / max(current_batch, 1)
        # })


# ----------------------------- #
#      Save the Predictions     #
# ----------------------------- #

assert len(sentiment_words_dict) <= len(SENTIMENT_CLASSES), print(len(sentiment_words_dict), len(SENTIMENT_CLASSES))
for sentiment_index, sentiment_words in sentiment_words_dict.items():
    cleaned_words = [word.strip(',.').lower() for word in sentiment_words]
    word_counts = defaultdict(int)
    for word in cleaned_words:
        word_counts[word] += 1
    word_count_sorted = dict(sorted(word_counts.items(), key=lambda x: -x[1]))
    sentiment_words_dict[sentiment_index] = word_count_sorted

with open(OUTPUT_PATH, 'w') as json_file:
    json.dump(sentiment_words_dict, json_file)
