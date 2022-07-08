"""
Modelling
"""
import logging
import math
from typing import *

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertEncoder,
    BertPooler,
    BaseModelOutputWithPoolingAndCrossAttentions,
)


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Positional embeddings
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        # half_dim shape
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # outer product (batch, 1) x (1, half_dim) -> (batch x half_dim)
        embeddings = time[:, None] * embeddings[None, :]
        # sin and cosine embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class BertForDiffusion(BertPreTrainedModel, pl.LightningModule):
    """
    BERT designed to be used with continuous inputs instead of tokens

    Reference: https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/bert/modeling_bert.py#L870
    """

    def __init__(self, config, add_pooling_layer: bool = False) -> None:
        """
        dim should be the dimension of the inputs
        """
        super().__init__(config)
        self.config = config

        # Needed to project the low dimensional input to hidden dim
        self.inputs_to_hidden_dim = nn.Linear(
            in_features=4, out_features=config.hidden_size
        )
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Set up the network to project token representation to our four outputs
        self.token_decoder = nn.Linear(config.hidden_size, 4)

        # Set up the time embedder
        self.time_embed = SinusoidalPositionEmbeddings(config.hidden_size)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        raise NotImplementedError

    def set_input_embeddings(self, value: nn.Module):
        raise NotImplementedError()

    def forward(
        self,
        inputs: torch.Tensor,
        timestep: torch.Tensor,  # Tensor of shape batch_length with time indices
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if inputs is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif inputs is not None:
            input_shape = inputs.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length, *_ = input_shape
        logging.debug(f"Detected batch {batch_size} and seq length {seq_length}")
        device = inputs.device if inputs is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device=device
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            raise NotImplementedError
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        inputs_upscaled = self.inputs_to_hidden_dim(inputs)  # Batch * seq_len * dim
        # timestep is (batch, 1), squeeze to (batch,)
        # embedding gets to (batch, embed_dim) -> unsqueee to (batch, 1, dim)
        time_encoded = self.time_embed(timestep.squeeze(dim=1)).unsqueeze(1)
        inputs_with_time = inputs_upscaled + time_encoded
        encoder_outputs = self.encoder(
            inputs_with_time,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        per_token_decoded = self.token_decoder(sequence_output)
        return per_token_decoded

    def training_step(self, batch, batch_idx):
        """
        Training step
        """
        known_noise = batch["known_noise"]
        predicted_noise = self.forward(
            batch["corrupted"], batch["t"], attention_mask=batch["attn_mask"]
        )

        unmask_idx = torch.where(batch["attn_mask"])
        loss = F.smooth_l1_loss(known_noise[unmask_idx], predicted_noise[unmask_idx])
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step
        """
        known_noise = batch["known_noise"]
        predicted_noise = self.forward(
            batch["corrupted"], batch["t"], attention_mask=batch["attn_mask"]
        )

        unmask_idx = torch.where(batch["attn_mask"])
        loss = F.smooth_l1_loss(known_noise[unmask_idx], predicted_noise[unmask_idx])
        self.log("val_loss", loss)

    def configure_optimizers(self):
        """
        Return optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def main():
    """on the fly testing"""
    import datasets
    from torch.utils.data.dataloader import default_collate

    clean_dset = datasets.CathConsecutiveAnglesDataset(toy=True)
    noised_dset = datasets.NoisedAnglesDataset(clean_dset)
    torch.utils.data.dataloader.default_collate
    x = default_collate([noised_dset[i] for i in range(8)])
    print(x["corrupted"].shape, x["corrupted"].dtype)
    print(x["t"].shape)

    # Create model
    # device = torch.device("cuda")
    model = BertForDiffusion(
        BertConfig(hidden_size=144, position_embedding_type="relative_key_query")
    )
    # print(model)
    y = model.forward(x["corrupted"], x["t"].squeeze())
    print(y.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
