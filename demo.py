import torch
import torch.nn.functional as F
import torch_npu                              # 适配 npu
from torch_npu.contrib import transfer_to_npu # 适配 npu
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from models.qwen2_vl import Qwen2VLRetForConditionalGeneration
from models.qwen2_vl_finetune import Qwen2VLRetFinetuneForConditionalGeneration
from tqdm import tqdm
import time

def add_embed_token(tokenizer, model, emb_token: str = "<emb>") -> None:
    """Add a new embedding token to the tokenizer and resize model embeddings.
    :param tokenizer: The tokenizer associated with the model.
    :param model: The Qwen2VL model to modify.
    :param emb_token: The new embedding token to add. Defaults to "<emb>".
    """   
    emb_tokens = [emb_token]
    num_new_tokens = tokenizer.add_tokens(emb_tokens)
    assert len(emb_tokens) == num_new_tokens
    model.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids(emb_token)
    if emb_token == "<instruction_start>":
        model.config.instruction_start_token_id = token_id
    elif emb_token == "<instruction_end>":
        model.config.instruction_end_token_id = token_id
    else:
        model.config.emb_token_id = token_id


def qwen2vl_process(messages: list, processor) -> dict:
    """Process messages into model inputs using the Qwen2VL processor.
    :param messages: List of message dictionaries containing text and media content.
    :param processor: The Qwen2VL processor for preprocessing inputs.

    :return: Dictionary of preprocessed inputs ready for model inference.
    """
    # Apply chat template to format messages
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]

    # Process vision information (images/videos) from messages
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to("npu:0")


# Load model and processor
model = Qwen2VLRetForConditionalGeneration.from_pretrained("checkpoints/U-MARVEL-Qwen2VL-7B-Instruct", torch_dtype=torch.bfloat16).to("npu:0")
# model = Qwen2VLRetFinetuneForConditionalGeneration.from_pretrained("checkpoints/U-MARVEL-Qwen2VL-7B-Instruct", torch_dtype=torch.bfloat16).to("npu:0")
processor = AutoProcessor.from_pretrained("checkpoints/U-MARVEL-Qwen2VL-7B-Instruct")
tokenizer = processor.tokenizer 
model.mean_pooling = True
model.use_bi_atten = True
model.use_latent_atten = False
model.use_instruction_mask = True

# Add embedding token to tokenizer and model
add_embed_token(tokenizer, model)
if model.use_instruction_mask:
    add_embed_token(tokenizer, model, emb_token="<instruction_start>")
    add_embed_token(tokenizer, model, emb_token="<instruction_end>")
    
# Define input messages for image and text queries
image_message = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "demo.jpeg"},
            {
                "type": "text",
                "text": "Find an image caption describing the following everyday image.",
            },
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
]

text_message1 = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "a dog and a woman are playing on the bench.\n",
            }
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
]

text_message2 = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "a dog."}
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
]

# Prepare batches of messages
query_messages = [image_message]
cand_messages = [text_message1,text_message2]

# Process inputs for model
query_inputs = qwen2vl_process(query_messages, processor)
query_inputs["labels"] = query_inputs["input_ids"].clone()
query_inputs["labels"][query_inputs["labels"] == tokenizer.pad_token_id] = -100


cand_inputs = qwen2vl_process(cand_messages, processor)
cand_inputs["labels"] = cand_inputs["input_ids"].clone()
cand_inputs["labels"][cand_inputs["labels"] == tokenizer.pad_token_id] = -100
        
        
# Extract embeddings with no gradient computation
with torch.no_grad():
    query_embeds = model(**query_inputs,inference=True).cpu()
    cand_embeds = model(**cand_inputs,inference=True).cpu()
# Normalize embeddings to unit vectors
query_embeds = F.normalize(query_embeds,dim=-1)
cand_embeds = F.normalize(cand_embeds,dim=-1)

print("Query Embeddings Shape:", query_embeds.shape)
print("Candidate Embeddings Shape:", cand_embeds.shape)
# Compute cosine similarity between query and candidate embeddings
similarity = query_embeds @ cand_embeds.t()
print(similarity)
        
