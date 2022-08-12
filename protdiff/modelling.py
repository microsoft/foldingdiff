"""
Modelling
"""
import os
import json
import inspect
import logging
import math
import functools
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
)

import losses
import utils


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    Built primarily for score-based models.

    Source:
    https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
    """

    def __init__(self, embed_dim: int, scale: float = 2 * torch.pi):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor):
        """
        takes as input the time vector and returns the time encoding
        time (x): (batch_size, )
        output  : (batch_size, embed_dim) 
        """
        x = x.squeeze()
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        embed = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return embed


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Positional embeddings
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
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


class PositionalEncoding(nn.Module):
    """
    Positional embedding for BERT.
    Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        assert len(x.shape) == 3
        orig_shape = x.shape
        # x is a tensor of shape (batch_size, seq_len, embedding_dim)
        # permute to be (seq_len, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)
        x += self.pe[: x.size(0)]
        # permute back to (batch_size, seq_len, embedding_dim)
        x = x.permute(1, 0, 2)
        assert x.shape == orig_shape, f"{x.shape} != {orig_shape}"
        return self.dropout(x)


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self, input_embeds: torch.Tensor, position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        assert position_ids is not None, "`position_ids` must be defined"
        embeddings = input_embeds
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertForDiffusion(BertPreTrainedModel, pl.LightningModule):
    """
    BERT designed to be used with continuous inputs instead of tokens

    Reference: https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/bert/modeling_bert.py#L870
    """

    loss_fn_dict = {
        "huber": F.smooth_l1_loss,
        "radian_l1": [
            F.smooth_l1_loss,
            losses.radian_l1_loss,
            losses.radian_l1_loss,
            losses.radian_l1_loss,
        ],
        "radian_l1_smooth": [
            F.smooth_l1_loss,
            functools.partial(losses.radian_smooth_l1_loss, beta=torch.pi / 10),
            functools.partial(losses.radian_smooth_l1_loss, beta=torch.pi / 10),
            functools.partial(losses.radian_smooth_l1_loss, beta=torch.pi / 10),
        ],
    }

    def __init__(
        self,
        config,
        n_inputs: int = 4,
        time_encoding: Literal["gaussian_fourier", "sinusoidal"] = "sinusoidal",
        lr: float = 1e-4,
        loss: Union[
            Callable, Literal["huber", "radian_l1", "radian_l1_smooth"]
        ] = "huber",
        l2: float = 0.0,
        l1: float = 0.0,
        circle_reg: float = 0.0,
        min_epochs: int = 1,
        steps_per_epoch: int = 250,  # Dummy value
        lr_scheduler: Optional[Literal["OneCycleLR"]] = None,
        write_preds_to_dir: Optional[str] = None,
    ) -> None:
        """
        dim should be the dimension of the inputs
        """
        super().__init__(config)
        self.config = config
        if self.config.is_decoder:
            raise NotImplementedError
        self.n_inputs = n_inputs

        # Store information about leraning rates and loss
        self.learning_rate = lr
        # loss function is either a callable or a list of callables
        self.loss_func = self.loss_fn_dict[loss] if isinstance(loss, str) else loss
        logging.info(f"Using loss: {self.loss_func}")
        self.l1_lambda = l1
        self.l2_lambda = l2
        self.circle_lambda = circle_reg
        self.min_epochs = min_epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr_scheduler = lr_scheduler

        # Needed to project the low dimensional input to hidden dim
        self.inputs_to_hidden_dim = nn.Linear(
            in_features=n_inputs, out_features=config.hidden_size
        )
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        # Set up the network to project token representation to our four outputs
        self.token_decoder = nn.Linear(config.hidden_size, n_inputs)

        # Set up the time embedder
        if time_encoding == "gaussian_fourier":
            self.time_embed = GaussianFourierProjection(config.hidden_size)
        elif time_encoding == "sinusoidal":
            self.time_embed = SinusoidalPositionEmbeddings(config.hidden_size)
        else:
            raise ValueError(f"Unknown time encoding: {time_encoding}")
        logging.info(f"Using time embedding: {self.time_embed}")

        # Initialize weights and apply final processing
        self.init_weights()

        # Set up the output directory for writing predictions
        self.write_preds_to_dir = write_preds_to_dir
        self.write_preds_counter = 0
        if self.write_preds_to_dir:
            os.makedirs(self.write_preds_to_dir, exist_ok=True)

    def forward(
        self,
        inputs: torch.Tensor,
        timestep: torch.Tensor,  # Tensor of shape batch_length with time indices
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
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

        input_shape = inputs.size()
        batch_size, seq_length, *_ = input_shape
        logging.debug(f"Detected batch {batch_size} and seq length {seq_length}")
        device = inputs.device

        assert attention_mask is not None

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device=device
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            torch.ones(size=(self.config.num_attention_heads,)).to(device),
            self.config.num_hidden_layers,
        )

        assert len(inputs.shape) == 3  # batch_size, seq_length, features
        inputs_upscaled = self.inputs_to_hidden_dim(inputs)  # Batch * seq_len * dim

        # Pass through embeddings
        inputs_upscaled = self.embeddings(inputs_upscaled, position_ids=position_ids)

        # timestep is (batch, 1), squeeze to (batch,)
        # embedding gets to (batch, embed_dim) -> unsqueee to (batch, 1, dim)
        time_encoded = self.time_embed(timestep.squeeze(dim=-1)).unsqueeze(1)
        inputs_with_time = inputs_upscaled + time_encoded
        encoder_outputs = self.encoder(
            inputs_with_time,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        per_token_decoded = self.token_decoder(sequence_output)
        return per_token_decoded

    def _get_loss_terms(
        self, batch, write_preds: Optional[str] = None
    ) -> List[torch.Tensor]:
        """
        Returns the loss terms for the model. Length of the returned list
        is equivalent to the number of features we are fitting to.
        """
        known_noise = batch["known_noise"]
        predicted_noise = self.forward(
            batch["corrupted"],
            batch["t"],
            attention_mask=batch["attn_mask"],
            position_ids=batch["position_ids"],
        )
        assert (
            known_noise.shape == predicted_noise.shape
        ), f"{known_noise.shape} != {predicted_noise.shape}"

        # Indexes into batch then indices along sequence length
        # attn_mask has shape (batch, seq_len) --> where gives back
        # two lists of values, one for each dimension
        # known_noise has shape (batch, seq_len, num_fts)
        unmask_idx = torch.where(batch["attn_mask"])
        assert len(unmask_idx) == 2
        loss_terms = []
        for i in range(known_noise.shape[-1]):
            loss_fn = (
                self.loss_func[i]
                if isinstance(self.loss_func, list)
                else self.loss_func
            )
            logging.debug(f"Using loss function {loss_fn}")
            # Determine whether the loss accepts circle_penalty
            # https://stackoverflow.com/questions/23228664/how-to-check-which-arguments-a-function-method-takes
            loss_args = inspect.getfullargspec(loss_fn)
            if (
                "circle_penalty" in loss_args.args
                or "circle_penalty" in loss_args.kwonlyargs
            ):
                logging.debug(f"Loss function {loss_fn} accepts circle_penalty")
                l = loss_fn(
                    predicted_noise[unmask_idx[0], unmask_idx[1], i],
                    known_noise[unmask_idx[0], unmask_idx[1], i],
                    circle_penalty=self.circle_lambda,
                )
            else:
                logging.debug(f"Loss function {loss_fn} does not accept circle_penalty")
                l = loss_fn(
                    predicted_noise[unmask_idx[0], unmask_idx[1], i],
                    known_noise[unmask_idx[0], unmask_idx[1], i],
                )
            loss_terms.append(l)

        if write_preds is not None:
            with open(write_preds, "w") as f:
                d_to_write = {
                    "known_noise": known_noise.cpu().numpy().tolist(),
                    "predicted_noise": predicted_noise.cpu().numpy().tolist(),
                    "attn_mask": batch["attn_mask"].cpu().numpy().tolist(),
                    "losses": [l.item() for l in loss_terms],
                }
                json.dump(d_to_write, f)

        return loss_terms

    def training_step(self, batch, batch_idx):
        """
        Training step
        """
        loss_terms = self._get_loss_terms(batch)
        avg_loss = torch.mean(torch.stack(loss_terms))

        # L1 loss implementation
        if self.l1_lambda > 0:
            l1_penalty = sum(torch.linalg.norm(p, 1) for p in self.parameters())
            avg_loss += self.l1_lambda * l1_penalty

        self.log("train_loss", avg_loss)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step
        """
        with torch.no_grad():
            loss_terms = self._get_loss_terms(
                batch,
                write_preds=os.path.join(
                    self.write_preds_to_dir, f"{self.write_preds_counter}_preds.json"
                ),
            )
            self.write_preds_counter += 1

        # Log each of the loss terms
        for val_name, val in zip(["bond_dist", "omega", "theta", "phi"], loss_terms):
            self.log(f"val_loss_{val_name}", val)

        avg_loss = torch.mean(torch.stack(loss_terms))
        self.log("val_loss", avg_loss)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Return optimizer. Limited support for some optimizers
        """
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.l2_lambda,
        )
        retval = {"optimizer": optim}
        if self.lr_scheduler:
            if self.lr_scheduler == "OneCycleLR":
                retval["lr_scheduler"] = {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        optim,
                        max_lr=1e-2,
                        epochs=self.min_epochs,
                        steps_per_epoch=self.steps_per_epoch,
                    ),
                    "monitor": "val_loss",
                    "frequency": 1,
                    "interval": "step",
                }
            else:
                raise ValueError(f"Unknown lr scheduler {self.lr_scheduler}")
        logging.info(f"Using optimizer {retval}")
        return retval


class BertDenoiserEncoderModel(pl.LightningModule):
    """
    Self implementation. Make sure that we know every bit of what goes into here so
    there's no more issues
    """

    loss_fn_dict = {
        "huber": F.smooth_l1_loss,
        "radian_l1": [
            F.smooth_l1_loss,
            losses.radian_l1_loss,
            losses.radian_l1_loss,
            losses.radian_l1_loss,
        ],
        "radian_l1_smooth": [
            F.smooth_l1_loss,
            functools.partial(losses.radian_smooth_l1_loss, beta=torch.pi / 10),
            functools.partial(losses.radian_smooth_l1_loss, beta=torch.pi / 10),
            functools.partial(losses.radian_smooth_l1_loss, beta=torch.pi / 10),
        ],
    }

    def __init__(
        self,
        n_inputs: int = 4,
        d_model: int = 256,
        num_layers: int = 6,
        intermediate_size: int = 512,
        max_seq_len: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        time_encoding: Literal["gaussian_fourier", "sinusoidal"] = "gaussian_fourier",
        loss: Union[
            Callable, Literal["huber", "radian_l1", "radian_l1_smooth"]
        ] = "huber",
        lr: float = 1e-4,
        l2: float = 0.0,
        l1: float = 0.0,
        circle_reg: float = 0.0,
        min_epochs: int = 500,
        steps_per_epoch: int = 100,  # Dummy value
        lr_scheduler: Optional[Literal["OneCycleLR"]] = None,
        write_preds_to_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.max_seq_len = max_seq_len
        self.learning_rate = lr
        self.l2_lambda = l2
        self.l1_lambda = l1
        self.circ_lambda = circle_reg
        if self.circ_lambda > 0:
            raise NotImplementedError
        self.min_epochs = min_epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr_scheduler = lr_scheduler

        self.loss_func = self.loss_fn_dict[loss] if isinstance(loss, str) else loss
        logging.info(f"Using loss: {self.loss_func}")

        # Define the positional embedding. Called as self.pos_encoder(x) and
        # returns the input + the positional embedding
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=self.max_seq_len, dropout=dropout
        )

        # Define the time embedding
        if time_encoding == "gaussian_fourier":
            self.time_encoder = GaussianFourierProjection(d_model)
        elif time_encoding == "sinusoidal":
            self.time_encoder = SinusoidalPositionEmbeddings(d_model)
        else:
            raise ValueError(f"Unknown time encoding {time_encoding}")
        logging.info(f"Time encoding: {self.time_encoder}")

        self.num_layers = num_layers
        self.d_model = d_model
        self.dropout = dropout
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.src_proj = nn.Linear(n_inputs, d_model)
        self.tgt_out = nn.Linear(d_model, n_inputs)

        # Define the transformer model itself
        # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
        self.transformer = self.get_transformer()

        self._init_weights()

        self.write_preds_to_dir = write_preds_to_dir
        self.write_preds_counter = 0
        if self.write_preds_to_dir:
            os.makedirs(self.write_preds_to_dir, exist_ok=True)

    def get_transformer(self) -> nn.Module:
        """
        Return the transformer model. Allows for easy overriding of the
        transformer aspect of the model for alternative architectures
        """
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.intermediate_size,
            dropout=self.dropout,
            activation="gelu",
            layer_norm_eps=1e-5,
            batch_first=False,  # Must do a permute to get batch first
            norm_first=True,
        )
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder
        encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)
        return encoder

    def _init_weights(self) -> None:
        # Initialize transformer with xavier uniform
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Upscale the features to (N, S, E) = (batch, seq_len, emb_size)
        x_upscaled = self.src_proj(x)

        # Add positional embeddings
        src_with_pos = self.pos_encoder(x_upscaled)

        # Add time embeddings
        # time_embed shape (batch, n_features) --> (batch, 1, n_features)
        time_embed = self.time_encoder(timestep.squeeze(1)).unsqueeze(1)
        assert time_embed.shape == (x.shape[0], 1, self.d_model)
        # src_with_pos shape (batch, seq_len, n_features)
        src_with_pos_time = src_with_pos + time_embed

        # Generate the src mask, follows https://pytorch.org/tutorials/beginner/translation_transformer.html
        # True --> NOT allowed to attend
        # False --> allowed to attend
        # Generate a vector of False
        src_mask = (
            torch.zeros((self.max_seq_len, self.max_seq_len))
            .type(torch.bool)
            .to(x.device)
        )
        # Feed through transformer
        # shape (batch, seq_len, d_model) --> (seq_len, batch, d_model) --> (batch, seq_len, d_model)
        decoded = self.transformer(
            src_with_pos_time.permute(1, 0, 2),
            mask=src_mask,
            src_key_padding_mask=attn_mask,
        ).permute(1, 0, 2)

        # Decode to targets
        out = self.tgt_out(decoded)
        assert out.shape == x.shape
        return out

    def ensure_mask_fmt(self, mask: torch.Tensor) -> torch.BoolTensor:
        """
        Ensure that the mask is given in the correct format (i.e., a True
        value indicates masked and a False indicates not masked). This is
        required because HuggingFace transformers use the opposite where
        a 1/True value indicates a position to be attended and 0/False
        indicates a position that is masked
        """
        assert torch.all(mask >= 0) and torch.all(mask <= 1)
        first_item = mask.flatten()[0]
        # if the first item is a 1.0 then we know that we have received
        # huggingface standard where 1 = attended. Flip to be 0 = attended
        if torch.isclose(first_item, torch.ones_like(first_item)):
            flipped_mask = ~(mask.bool())
            assert torch.all(
                torch.sum(mask) == torch.numel(flipped_mask) - torch.sum(flipped_mask)
            )
            assert torch.all(flipped_mask[torch.where(mask)] == False)
            return flipped_mask.bool()
        return mask.bool()

    def _get_loss_terms(self, batch, write_preds: Optional[str] = None) -> torch.Tensor:
        """
        Gets the loss terms for the model
        """
        known_noise = batch["known_noise"]
        corrupted = batch["corrupted"]
        # Make sure the attention mask is False for unmasked
        attn_mask = self.ensure_mask_fmt(batch["attn_mask"])
        assert (
            attn_mask.dtype == torch.bool
        ), f"{attn_mask} is not boolean - {attn_mask.dtype}"

        predicted_noise = self.forward(
            corrupted, timestep=batch["t"], attn_mask=attn_mask
        )

        # Under pytorch convention, 0 = not masked
        unmask_idx = torch.where(attn_mask == 0)
        assert len(unmask_idx) == 2
        loss_terms = []
        for i in range(known_noise.shape[-1]):
            loss_fn = (
                self.loss_func[i]
                if isinstance(self.loss_func, list)
                else self.loss_func
            )
            logging.debug(f"Using loss function {loss_fn}")

            l = loss_fn(
                predicted_noise[unmask_idx[0], unmask_idx[1], i],
                known_noise[unmask_idx[0], unmask_idx[1], i],
            )
            loss_terms.append(l)

        if write_preds is not None:
            with open(write_preds, "w") as f:
                d_to_write = {
                    "known_noise": known_noise.cpu().numpy().tolist(),
                    "predicted_noise": predicted_noise.cpu().numpy().tolist(),
                    "attn_mask": attn_mask.cpu().numpy().tolist(),
                    "losses": [l.item() for l in loss_terms],
                }
                json.dump(d_to_write, f)

        return torch.stack(loss_terms)

    def training_step(self, batch, batch_idx):
        """
        Training step for the model
        """
        loss = self._get_loss_terms(batch)
        avg_loss = torch.mean(loss)

        # L1 regularization
        if self.l1_lambda > 0:
            l1_penalty = sum(torch.linalg.norm(p, 1) for p in self.parameters())
            self.log("l1_penalty", l1_penalty)
            avg_loss += self.l1_lambda * l1_penalty

        for loss_name, loss_val in zip(["bond_dist", "omega", "theta", "phi"], loss):
            self.log(f"train_{loss_name}", loss_val)
        self.log("train_loss", avg_loss)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss_terms = self._get_loss_terms(
                batch,
                write_preds=os.path.join(
                    self.write_preds_to_dir, f"{self.write_preds_counter}_preds.json"
                ),
            )
            self.write_preds_counter += 1

        avg_loss = torch.mean(loss_terms)

        for loss_name, loss_val in zip(
            ["bond_dist", "omega", "theta", "phi"], loss_terms
        ):
            self.log(f"val_{loss_name}", loss_val)
        self.log("val_loss", avg_loss)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        References:
        * https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html
        * https://pytorch.org/docs/stable/optim.html
        """
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.l2_lambda,
        )
        retval = {"optimizer": optim}
        if self.lr_scheduler:
            if self.lr_scheduler == "OneCycleLR":
                retval["lr_scheduler"] = {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        optim,
                        max_lr=1e-2,
                        epochs=self.min_epochs,
                        steps_per_epoch=self.steps_per_epoch,
                    ),
                    "monitor": "val_loss",
                    "frequency": 1,
                    "interval": "step",
                }
            else:
                raise ValueError(f"Unknown lr scheduler {self.lr_scheduler}")
        logging.info(f"Using optimizer {retval}")
        return retval


class BertDenoiserSeq2SeqModel(BertDenoiserEncoderModel):
    """
    Use a seq2seq model instead of a encoder only transformer
    """

    def get_transformer(self) -> nn.Module:
        nn.Transformer(
            d_model=self.d_model,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            norm_first=False,
            batch_first=True,
        )
        raise NotImplementedError

    def get_causal_tgt_mask(
        self, tgt_seq_len: Optional[int] = None
    ) -> torch.BoolTensor:
        """
        Get a causal mask for target sequence where each row allows only
        the next token to be seen. This is important because otherwise
        the decoder can simply pass through the known target sequence.
        Example output:
        # [F, T, T, T]
        # [F, F, T, T]
        # [F, F, F, T]
        # [F, F, F, F]
        """
        # Lower triangular matrix
        if tgt_seq_len is None:
            tgt_seq_len = self.max_seq_len
        mask = ~torch.tril(torch.ones(tgt_seq_len, tgt_seq_len) == 1).bool()
        # If a BoolTensor is provided, positions with True are not allowed to attend
        # while False values will be unchanged (i.e., True => masked)
        # If a FloatTensor is provided, it will be added to the attention weight
        return mask


def main():
    """on the fly testing"""
    import datasets
    from torch.utils.data import default_collate

    clean_dset = datasets.CathConsecutiveAnglesDataset(toy=True)
    noised_dset = datasets.NoisedAnglesDataset(clean_dset, "angles")
    for k, v in noised_dset[0].items():
        print(k, v.shape)
    x = default_collate([noised_dset[i] for i in range(8)])

    # # Create model
    # device = torch.device("cuda")
    model = BertDenoiserEncoderModel()
    # print(model)
    y = model.forward(x["corrupted"], x["t"].squeeze())
    print(y.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
