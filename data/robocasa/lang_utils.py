import os
import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection, BertModel
from llm2vec import LLM2Vec
from typing import Literal


class CLIPLangEncoder:
    def __init__(self, device):
        os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock
        model_variant = "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32"
        self.device = device
        self.lang_emb_model = CLIPTextModelWithProjection.from_pretrained(
            model_variant,
        ).to(device).eval()
        self.tz = AutoTokenizer.from_pretrained(model_variant, TOKENIZERS_PARALLELISM=True)

    def get_lang_emb(self, lang):
        if lang is None:
            return None
        
        with torch.no_grad():
            tokens = self.tz(
                text=lang,                   # the sentence to be encoded
                add_special_tokens=True,             # Add [CLS] and [SEP]
                # max_length=25,  # maximum length of a sentence
                padding="max_length",
                return_attention_mask=True,        # Generate the attention mask
                return_tensors="pt",               # ask the function to return PyTorch tensors
            ).to(self.device)

            lang_emb = self.lang_emb_model(**tokens)['text_embeds'].detach()
        
        # check if input is batched or single string
        if isinstance(lang, str):
            lang_emb = lang_emb[0]

        return lang_emb


class BERTLangEncoder:
    def __init__(self, device):
        os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock
        model_variant = "google-bert/bert-base-uncased" 
        self.device = device
        self.lang_emb_model = BertModel.from_pretrained(
            model_variant,
        ).to(device).eval()
        self.tz = AutoTokenizer.from_pretrained(model_variant, TOKENIZERS_PARALLELISM=True)

    def get_lang_emb(self, lang):
        if lang is None:
            return None
        
        with torch.no_grad():
            tokens = self.tz(
                text=lang,                   # the sentence to be encoded
                add_special_tokens=True,             # Add [CLS] and [SEP]
                # max_length=25,  # maximum length of a sentence
                padding="max_length",
                return_attention_mask=True,        # Generate the attention mask
                return_tensors="pt",               # ask the function to return PyTorch tensors
            ).to(self.device)

            lang_emb = self.lang_emb_model(**tokens)['last_hidden_state'][:, 0].detach() 
        
        # check if input is batched or single string
        if isinstance(lang, str):
            lang_emb = lang_emb[0]

        return lang_emb


class LLM2VecLangEncoder:
    def __init__(self, 
                 model: Literal["mistral", "llama2", "llama3", "sheared_llama"], 
                 mode: Literal["word", "sentence_unsupervised", "sentence_supervised"], 
                 device,
                 instruction: str = None
                 ):
        os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock
        self.device = device
        self.instruction = instruction

        base_model = {"mistral": "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp", 
                      "llama2": "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp",
                      "llama3": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                      "sheared_llama": "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"}
        peft = {"word": "", "sentence_unsupervised": "-unsup-simcse", "sentence_supervised": "-supervised"}

        self.lang_emb_model = LLM2Vec.from_pretrained(
            base_model[model],
            peft_model_name_or_path=base_model[model] + peft[mode],
            device_map=device,
            torch_dtype=torch.bfloat16,
        ).eval()

    def get_lang_emb(self, lang):
        if lang is None:
            return None
        
        if self.instruction is not None:
            if isinstance(lang, str):
                lang = [self.instruction, lang]
            else:
                lang = [[self.instruction, l] for l in lang]
        
        with torch.no_grad():
            lang_emb = self.lang_emb_model.encode(lang).detach() 
        
        # check if input is batched or single string
        if isinstance(lang, str):
            lang_emb = lang_emb[0]

        return lang_emb


def language_encoder_factory(model: Literal["clip", "bert", "mistral", "llama2", "llama3", "sheared_llama"], device):
    hf_dict = {"clip": CLIPLangEncoder, "bert": BERTLangEncoder}
    llm_list = ["mistral", "llama2", "llama3", "sheared_llama"]

    if model in llm_list:
        return LLM2VecLangEncoder(model, mode="sentence_supervised", device=device, instruction=None)           # E.g. instruction="In the context of a robotic arm executing instructions, retrieve relevant information to execute the command:"
    
    return hf_dict[model](device)