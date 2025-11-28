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

class Patchilizer:
    def __init__(self, stream=PATCH_STREAM):
        self.stream = stream
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.special_token_id = 0

    def split_bars(self, body_lines):
        """
        Split a body of music into individual bars.
        """
        new_bars = []
        try:
            for line in body_lines:
                line_bars = re.split(self.regexPattern, line)
                line_bars = list(filter(None, line_bars))
                new_line_bars = []

                if len(line_bars) == 1:
                    new_line_bars = line_bars
                else:
                    if line_bars[0] in self.delimiters:
                        new_line_bars = [line_bars[i] + line_bars[i + 1] for i in range(0, len(line_bars), 2)]
                    else:
                        new_line_bars = [line_bars[0]] + [line_bars[i] + line_bars[i + 1] for i in
                                                          range(1, len(line_bars), 2)]
                    if 'V' not in new_line_bars[-1]:
                        new_line_bars[-2] += new_line_bars[-1]
                        new_line_bars = new_line_bars[:-1]
                new_bars += new_line_bars
        except:
            pass

        return new_bars

    def split_patches(self, abc_text, patch_size=PATCH_SIZE, generate_last=False):
        if not generate_last and len(abc_text) % patch_size != 0:
            abc_text += chr(self.eos_token_id)
        patches = [abc_text[i: i + patch_size] for i in range(0, len(abc_text), patch_size)]
        return patches

    def patch2chars(self, patch):
        """
        Convert a patch into a bar.
        """
        bytes = ''
        for idx in patch:
            if idx == self.eos_token_id:
                break
            if idx < self.eos_token_id:
                pass
            bytes += chr(idx)
        return bytes

    def patchilize_metadata(self, metadata_lines):

        metadata_patches = []
        for line in metadata_lines:
            metadata_patches += self.split_patches(line)

        return metadata_patches

    def patchilize_tunebody(self, tunebody_lines, encode_mode='train'):

        tunebody_patches = []
        bars = self.split_bars(tunebody_lines)
        if encode_mode == 'train':
            for bar in bars:
                tunebody_patches += self.split_patches(bar)
        elif encode_mode == 'generate':
            for bar in bars[:-1]:
                tunebody_patches += self.split_patches(bar)
            tunebody_patches += self.split_patches(bars[-1], generate_last=True)

        return tunebody_patches

    def encode_train(self, abc_text, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=True,
                     cut=True):

        lines = abc_text.split('\n')
        lines = list(filter(None, lines))
        lines = [line + '\n' for line in lines]

        tunebody_index = -1
        for i, line in enumerate(lines):
            if line.startswith('[V:'):
                tunebody_index = i
                break

        metadata_lines = lines[: tunebody_index]
        tunebody_lines = lines[tunebody_index:]

        if self.stream:
            tunebody_lines = ['[r:' + str(line_index) + '/' + str(len(tunebody_lines) - line_index - 1) + ']' + line for
                              line_index, line in
                              enumerate(tunebody_lines)]  # [r:n/n]

        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='train')

        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)
            eos_patch = chr(self.bos_token_id) + chr(self.eos_token_id) * (patch_size - 1)

            metadata_patches = [bos_patch] + metadata_patches
            tunebody_patches = tunebody_patches + [eos_patch]

        if self.stream:
            if len(metadata_patches) + len(tunebody_patches) > patch_length:
                available_cut_indexes = [0] + [index + 1 for index, patch in enumerate(tunebody_patches) if
                                               '\n' in patch]
                line_index_for_cut_index = list(range(len(available_cut_indexes)))
                end_index = len(metadata_patches) + len(tunebody_patches) - patch_length
                biggest_index = bisect.bisect_left(available_cut_indexes, end_index)
                available_cut_indexes = available_cut_indexes[:biggest_index + 1]

                if len(available_cut_indexes) == 1:
                    choices = ['head']
                elif len(available_cut_indexes) == 2:
                    choices = ['head', 'tail']
                else:
                    choices = ['head', 'tail', 'middle']
                choice = random.choice(choices)
                if choice == 'head':
                    patches = metadata_patches + tunebody_patches[0:]
                else:
                    if choice == 'tail':
                        cut_index = len(available_cut_indexes) - 1
                    else:
                        cut_index = random.choice(range(1, len(available_cut_indexes) - 1))

                    line_index = line_index_for_cut_index[cut_index]
                    stream_tunebody_lines = tunebody_lines[line_index:]

                    stream_tunebody_patches = self.patchilize_tunebody(stream_tunebody_lines, encode_mode='train')
                    if add_special_patches:
                        stream_tunebody_patches = stream_tunebody_patches + [eos_patch]
                    patches = metadata_patches + stream_tunebody_patches
            else:
                patches = metadata_patches + tunebody_patches
        else:
            patches = metadata_patches + tunebody_patches

        if cut:
            patches = patches[: patch_length]
        else:
            pass

        # encode to ids
        id_patches = []
        for patch in patches:
            id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
            id_patches.append(id_patch)

        return id_patches

    def encode_generate(self, abc_code, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=True):

        lines = abc_code.split('\n')
        lines = list(filter(None, lines))

        tunebody_index = None
        for i, line in enumerate(lines):
            if line.startswith('[V:') or line.startswith('[r:'):
                tunebody_index = i
                break

        metadata_lines = lines[: tunebody_index]
        tunebody_lines = lines[tunebody_index:]

        metadata_lines = [line + '\n' for line in metadata_lines]
        if self.stream:
            if not abc_code.endswith('\n'):
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [
                    tunebody_lines[-1]]
            else:
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]
        else:
            tunebody_lines = [line + '\n' for line in tunebody_lines]

        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='generate')

        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)

            metadata_patches = [bos_patch] + metadata_patches

        patches = metadata_patches + tunebody_patches
        patches = patches[: patch_length]

        # encode to ids
        id_patches = []
        for patch in patches:
            if len(patch) < PATCH_SIZE and patch[-1] != chr(self.eos_token_id):
                id_patch = [ord(c) for c in patch]
            else:
                id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
            id_patches.append(id_patch)

        return id_patches

    def decode(self, patches):
        """
        Decode patches into music.
        """
        return ''.join(self.patch2chars(patch) for patch in patches)