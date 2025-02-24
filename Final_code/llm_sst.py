import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
from tqdm import tqdm

# ----------------------------- #
#        Configuration          #
# ----------------------------- #

# Path to the CSV data
DATA_PATH = '/mnt/petrelfs/chenlingjie/google-research/goemotions/data/test.tsv'
use_lora = True
# Paths to the tokenizer and model
TOKENIZER_PATH = "/mnt/hwfile/llm-safety/models/huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct"
ORIGINAL_MODEL_PATH = "/mnt/hwfile/llm-safety/models/huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_MODEL_PATH = "/mnt/petrelfs/chenlingjie/google-research/models/meta-llama/Meta-Llama-3.1-8B-Instruct-sst-1-epoch/checkpoint-2714"

# Output path for predictions
OUTPUT_PATH = 'goemotions_with_predictions__llama_3.1_8B_instruct_sst_finetuned.csv'

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
        "Classify the following text into one of the following sentiment classes: \n"
        f"{', '.join(SENTIMENT_CLASSES)}.\n\n"
        "Examples:\n"
        "1. 'That game hurt.' -> sadness\n"
        "2. 'You do right, if you don't care then fuck 'em!' -> neutral\n"
        "3. 'Man I love reddit' -> love\n"
        "4. 'that's adorable asf' -> amusement\n"
        "5. 'That is odd.' -> disappointment, disgust\n\n"
        f"Text: {text}\n"
        "You only need to output the sentiment contained in the given text. And most text only contains one sentiment.\n"
        "Sentiment:"
    )
    return prompt

# ----------------------------- #
#        Inference Loop         #
# ----------------------------- #

print("Starting sentiment classification...")
predictions = []
num_batches = (len(csv_data) + BATCH_SIZE - 1) // BATCH_SIZE
false_prediction_count = 0
duplex_prediction = 0

with tqdm(range(0, len(csv_data), BATCH_SIZE), total=num_batches, ncols=100) as pbar:
    for i in pbar:
        batch_texts = csv_data['text'][i:i+BATCH_SIZE].tolist()
        prompts = [create_prompt(text) for text in batch_texts]
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda')
        
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
        # breakpoint()
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for gen_text in generated_texts:
            predicted_result = [0] * len(SENTIMENT_CLASSES)
            # The expected format: "Classify the ... Sentiment: <label>"
            # We extract the part after 'Sentiment:'
            try:
                sentiment = gen_text.split("Sentiment:")[-1].strip().split('\n')[0].strip()
                sentiments_list = [s.strip().rstrip(',').rstrip('.') for s in sentiment.split(',') if s.strip()]
                for s in sentiments_list:
                    if s in SENTIMENT_CLASSES:
                        predicted_result[SENTIMENT_CLASSES.index(s)] = 1
                if sum(predicted_result) > 1:
                    duplex_prediction += 1
            except Exception as e:
                false_prediction_count += 1
            predictions.append(predicted_result)
        
        current_batch = i + BATCH_SIZE
        pbar.set_postfix({
            "None Prediction": false_prediction_count / max(current_batch, 1), 
            "Duplex Prediction": duplex_prediction / max(current_batch, 1)
        })


# ----------------------------- #
#      Save the Predictions     #
# ----------------------------- #

print("Adding predictions to the dataframe...")
csv_data['predicted_sentiment'] = predictions

print(f"Saving predictions to {OUTPUT_PATH}...")
csv_data.to_csv(OUTPUT_PATH, index=False)

print("Sentiment classification completed successfully!")
