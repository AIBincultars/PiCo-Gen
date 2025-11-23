import torch
import random
import bisect
import json
import re
import numpy as np
from config import *
from transformers import GPT2Model, GPT2LMHeadModel, LlamaModel, LlamaForCausalLM, PreTrainedModel
from samplings import top_p_sampling, top_k_sampling, temperature_sampling
from tokenizers import Tokenizer



class PatchLevelDecoder(PreTrainedModel):
    """
    A Patch-level Decoder model for generating patch features in an auto-regressive manner.
    It inherits PreTrainedModel from transformers.
    """

    def __init__(self, config):
        super().__init__(config)
        self.patch_embedding = torch.nn.Linear(PATCH_SIZE * 128, config.n_embd)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.base = GPT2Model(config)

    def forward(self,
                patches: torch.Tensor,
                masks=None) -> torch.Tensor:
        """
        The forward pass of the patch-level decoder model.
        :param patches: the patches to be encoded
        :param masks: the masks for the patches
        :return: the encoded patches
        """
        patches = torch.nn.functional.one_hot(patches, num_classes=128).to(self.dtype)
        patches = patches.reshape(len(patches), -1, PATCH_SIZE * (128))
        patches = self.patch_embedding(patches.to(self.device))

        if masks == None:
            return self.base(inputs_embeds=patches)
        else:
            return self.base(inputs_embeds=patches,
                             attention_mask=masks)


class CharLevelDecoder(PreTrainedModel):
    """
    A Char-level Decoder model for generating the chars within each patch in an auto-regressive manner
    based on the encoded patch features. It inherits PreTrainedModel from transformers.
    """

    def __init__(self, config):
        super().__init__(config)
        self.special_token_id = 0
        self.bos_token_id = 1

        self.base = GPT2LMHeadModel(config)

    def forward(self,
                encoded_patches: torch.Tensor,
                target_patches: torch.Tensor):
        """
        The forward pass of the char-level decoder model.
        :param encoded_patches: the encoded patches
        :param target_patches: the target patches
        :return: the output of the model
        """
        # preparing the labels for model training
        target_patches = torch.cat((torch.ones_like(target_patches[:, 0:1]) * self.bos_token_id, target_patches), dim=1)

        target_masks = target_patches == self.special_token_id
        labels = target_patches.clone().masked_fill_(target_masks, -100)

        # masking the labels for model training
        target_masks = torch.ones_like(labels)
        target_masks = target_masks.masked_fill_(labels == -100, 0)

        # select patches
        if PATCH_SAMPLING_BATCH_SIZE != 0 and PATCH_SAMPLING_BATCH_SIZE < target_patches.shape[0]:
            indices = list(range(len(target_patches)))
            random.shuffle(indices)
            selected_indices = sorted(indices[:PATCH_SAMPLING_BATCH_SIZE])

            target_patches = target_patches[selected_indices, :]
            target_masks = target_masks[selected_indices, :]
            encoded_patches = encoded_patches[selected_indices, :]

        # get input embeddings
        inputs_embeds = torch.nn.functional.embedding(target_patches, self.base.transformer.wte.weight)

        # concatenate the encoded patches with the input embeddings
        inputs_embeds = torch.cat((encoded_patches.unsqueeze(1), inputs_embeds[:, 1:, :]), dim=1)

        output = self.base(inputs_embeds=inputs_embeds,
                           attention_mask=target_masks,
                           labels=labels)

        return output

    def generate(self,
                 encoded_patch: torch.Tensor,
                 tokens: torch.Tensor):
        """
        The generate function for generating a patch based on the encoded patch and already generated tokens.
        :param encoded_patch: the encoded patch
        :param tokens: already generated tokens in the patch
        :return: the probability distribution of next token
        """
        encoded_patch = encoded_patch.reshape(1, 1, -1)
        tokens = tokens.reshape(1, -1)

        # Get input embeddings
        tokens = torch.nn.functional.embedding(tokens, self.base.transformer.wte.weight)

        # Concatenate the encoded patch with the input embeddings
        tokens = torch.cat((encoded_patch, tokens[:, 1:, :]), dim=1)

        # Get output from model
        outputs = self.base(inputs_embeds=tokens)

        # Get probabilities of next token
        probs = torch.nn.functional.softmax(outputs.logits.squeeze(0)[-1], dim=-1)

        return probs


def safe_normalize_probs(probs):
    epsilon = 1e-12
    probs = np.array(probs, dtype=np.float64)
    probs = np.where(np.isnan(probs) | (probs < 0), 0, probs)
    probs = probs + epsilon
    s = probs.sum()
    if s > 0:
        probs = probs / s
    else:
        probs = np.zeros_like(probs)
        probs[0] = 1.0
    return probs


class NotaGenLMHeadModel(PreTrainedModel):
    """
    NotaGen is a language model with a hierarchical structure.
    It includes a patch-level decoder and a char-level decoder.
    The patch-level decoder is used to generate patch features in an auto-regressive manner.
    The char-level decoder is used to generate the chars within each patch in an auto-regressive manner.
    It inherits PreTrainedModel from transformers.
    """

    def __init__(self, encoder_config, decoder_config):
        super().__init__(encoder_config)
        self.special_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.patch_level_decoder = PatchLevelDecoder(encoder_config)
        self.char_level_decoder = CharLevelDecoder(decoder_config)

    def forward(self,
                patches: torch.Tensor,
                masks: torch.Tensor):
        """
        The forward pass of the bGPT model.
        :param patches: the patches to be encoded
        :param masks: the masks for the patches
        :return: the decoded patches
        """
        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        encoded_patches = self.patch_level_decoder(patches, masks)["last_hidden_state"]

        left_shift_masks = masks * (masks.flip(1).cumsum(1).flip(1) > 1)
        masks[:, 0] = 0

        encoded_patches = encoded_patches[left_shift_masks == 1]
        patches = patches[masks == 1]

        return self.char_level_decoder(encoded_patches, patches)

    def generate(self,
                 patches: torch.Tensor,
                 top_k=0,
                 top_p=1,
                 temperature=1.0):
        """
        The generate function for generating patches based on patches.
        :param patches: the patches to be encoded
        :param top_k: the top k for sampling
        :param top_p: the top p for sampling
        :param temperature: the temperature for sampling
        :return: the generated patches
        """
        if patches.shape[-1] % PATCH_SIZE != 0:
            tokens = patches[:, :, -(patches.shape[-1] % PATCH_SIZE):].squeeze(0, 1)
            tokens = torch.cat((torch.tensor([self.bos_token_id], device=self.device), tokens), dim=-1)
            patches = patches[:, :, :-(patches.shape[-1] % PATCH_SIZE)]
        else:
            tokens = torch.tensor([self.bos_token_id], device=self.device)

        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        encoded_patches = self.patch_level_decoder(patches)["last_hidden_state"]
        generated_patch = []

        while True:
            prob = self.char_level_decoder.generate(encoded_patches[0][-1], tokens).cpu().detach().numpy()
            prob = safe_normalize_probs(prob)
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True)
            prob = safe_normalize_probs(prob)
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True)
            prob = safe_normalize_probs(prob)
            token = temperature_sampling(prob, temperature=temperature)
            char = chr(token)
            generated_patch.append(token)

            if len(tokens) >= PATCH_SIZE:
                break
            else:
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)

        return generated_patch