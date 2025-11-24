from collections import OrderedDict
from math import log
from typing import List, Tuple, Union

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal, kl_divergence
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

START = -1  # Start token value
PAD = -2  # Padding token value


def build_vocab(
    num_tables: int,
    num_aliases: int,
):
    table_alias = [(i, j + 1) for i in range(num_tables) for j in range(num_aliases)]
    ops = [1, 2, 3]  # Join operators
    special = [START, PAD]

    all_tokens: List[Union[int, Tuple[int, int]]] = special + ops + table_alias
    vocab = OrderedDict(zip(all_tokens, range(len(all_tokens))))
    rev_vocab = OrderedDict(zip(range(len(all_tokens)), all_tokens))
    return vocab, rev_vocab


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :]  # type: ignore
        return self.dropout(x)


class InfoTransformerVAE(L.LightningModule):
    def __init__(
        self,
        vocab: dict,
        rev_vocab: dict,
        bn_size: int = 2,
        d_enc: int = 128,
        d_dec: int = 128,
        d_neck: int = 128,
        kl_factor: float = 0.1,
        min_posterior_std: float = 1e-4,
        enc_nhead: int = 8,
        enc_dim_ff: int = 512,
        enc_dropout: float = 0.1,
        enc_num_layer: int = 6,
        dec_nhead: int = 8,
        dec_dim_ff: int = 256,
        dec_dropout: float = 0.1,
        dec_num_layer: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab = vocab
        self.rev_vocab = rev_vocab
        self.vocab_size = len(vocab)

        self.bn_size = bn_size
        self.d_model = d_dec
        self.d_neck = d_neck

        self.kl_factor = kl_factor

        self.min_posterior_std = min_posterior_std

        self.enc_tok_emb = nn.Embedding(self.vocab_size, embedding_dim=d_enc)
        self.enc_pos_enc = PositionalEncoding(d_enc, dropout=enc_dropout)

        self.dec_pos_enc = PositionalEncoding(self.d_model, dropout=enc_dropout)
        self.dec_tok_emb = nn.Embedding(self.vocab_size, embedding_dim=d_dec)
        self.dec_tok_unembed = nn.Linear(d_dec, self.vocab_size)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_enc,
                nhead=enc_nhead,
                dim_feedforward=enc_dim_ff,
                dropout=enc_dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=enc_num_layer,
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_dec,
                nhead=dec_nhead,
                dim_feedforward=dec_dim_ff,
                dropout=dec_dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=dec_num_layer,
        )

        d_latent = self.bn_size * d_neck
        self.enc_neck = nn.Linear(d_enc * self.bn_size, d_latent * 2)
        self.dec_neck = nn.Linear(d_latent, d_dec * self.bn_size)

        reset(self)

    def encode(self, tokens):
        embed = self.enc_tok_emb(tokens)
        embed = self.enc_pos_enc(embed)

        pad_mask = tokens == self.vocab[PAD]
        embed = self.encoder(embed, src_key_padding_mask=pad_mask)[:, : self.bn_size]
        embed = self.enc_neck(embed.flatten(1))

        mu, sigma = embed.chunk(2, dim=-1)
        sigma = F.softplus(sigma) + self.min_posterior_std

        return mu, sigma

    def decode(self, z, tokens):
        z = self.dec_neck(z).reshape(z.shape[0], self.bn_size, self.d_model)
        embed = self.dec_tok_emb(tokens)
        embed = self.dec_pos_enc(embed)

        decoding = self.decoder(
            tgt=embed,
            memory=z,
            tgt_mask=causal_mask(embed.shape[1], embed.dtype, embed.device),
            tgt_is_causal=True,
        )
        logits = self.dec_tok_unembed(decoding)

        return logits

    def forward(self, tokens):
        mu, sigma = self.encode(tokens)
        z = mu + sigma * torch.randn_like(mu)

        logits = self.decode(z, tokens)

        loss_tokens = tokens[:, 1:]
        loss_logits = logits[:, :-1]

        recon_loss = F.cross_entropy(loss_logits.permute(0, 2, 1), loss_tokens)
        kldiv = kl_divergence(Normal(mu, sigma), Normal(0, 1)).mean(dim=-1)

        loss = recon_loss + self.hparams.kl_factor * kldiv  # type: ignore
        loss = loss.mean()

        with torch.no_grad():
            preds = loss_logits.argmax(dim=-1)
            hits = loss_tokens == preds

            token_acc = hits.float().mean()
            string_acc = hits.all(dim=1).float().mean()
            sigma_mean = sigma.mean()

        return dict(
            loss=loss,
            z=z,
            recon_loss=recon_loss,
            kldiv=kldiv,
            recon_token_acc=token_acc,
            recon_string_acc=string_acc,
            sigma_mean=sigma_mean,
            mu=mu,
            sigma=sigma,
        )


class VAEModule(L.LightningModule):
    def __init__(
        self,
        vocab: dict,
        rev_vocab: dict,
        bn_size: int = 2,
        d_enc: int = 128,
        d_dec: int = 128,
        d_neck: int = 128,
        kl_factor: float = 0.1,
        min_posterior_std: float = 1e-4,
        enc_nhead: int = 8,
        enc_dim_ff: int = 512,
        enc_dropout: float = 0.1,
        enc_num_layer: int = 6,
        dec_nhead: int = 8,
        dec_dim_ff: int = 256,
        dec_dropout: float = 0.1,
        dec_num_layer: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = InfoTransformerVAE(**self.hparams)

        self.max_string_length = 90

        self.vocab = vocab
        self.rev_vocab = rev_vocab

    def forward(self, tokens: List[List[int]]):
        try:
            tokens = encode_batch(tokens, self.vocab).to(self.device)  # type: ignore
        except:  # noqa: E722
            msg = "Compressed encoding failed. Output from the VAE is likely being fed back into the VAE which can cause issues if the output is not a valid encoding."
            raise RuntimeError(msg)

        stats = self.model(tokens)
        return stats

    def training_step(self, batch, batch_idx):
        out = self(batch)

        out = {f"train/{k}": v for k, v in out.items() if k not in ("z", "mu", "sigma")}
        self.log_dict(out, prog_bar=True, logger=True)

        return out["train/loss"]

    def validation_step(self, batch, batch_idx):
        out = self(batch)

        self.metric.update(out["mu"], out["sigma"])
        self.log(
            "val/alive",
            self.metric,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        out = {f"val/{k}": v for k, v in out.items() if k not in ("z", "mu", "sigma")}
        self.log_dict(out, prog_bar=True, logger=True)

        return out["val/loss"]

    def configure_optimizers(self):  # type: ignore
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.99))
        lr_sched = LambdaLR(opt, lambda step: min(1.0, step / 2048))

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": lr_sched, "interval": "step", "frequency": 1},
        }

    @torch.no_grad()
    def sample(self, z: torch.Tensor):
        pr = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("medium")
        tr = self.training
        self.eval()

        if z.ndim == 1:
            z = z.unsqueeze(0)

        n = z.shape[0]
        z = z.reshape(n, self.hparams.bn_size * self.hparams.d_neck).to(dtype=torch.float32)  # type: ignore

        tokens = torch.full((n, 1), fill_value=self.vocab[START], device=self.device)
        while True:
            logits = self.model.decode(z, tokens)[:, -1:]
            sample = Categorical(logits=logits).sample()

            tokens = torch.hstack([tokens, sample])

            stop_mask = (tokens == self.vocab[PAD]).any(dim=-1).all()
            if stop_mask or tokens.shape[-1] > self.max_string_length:
                break

        tokens = decode_batch(tokens, self.rev_vocab)

        self.train(tr)
        torch.set_float32_matmul_precision(pr)
        return tokens


def causal_mask(sz: int, dtype: torch.dtype, device: torch.device):
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )


def reset(module: InfoTransformerVAE):
    for mod in module.modules():
        if isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, std=module.d_model**-0.5)
        elif isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

    for mod in module.modules():
        if isinstance(mod, nn.MultiheadAttention):
            mod._reset_parameters()


def encode_batch(batch: Union[List[List[int]], List[int]], vocab) -> torch.Tensor:
    """
    Re-encode the flat encoding to create tokens for table,alias pairs and operators
    3,1,2,2,3,2,1,3,1,3

    Breakdown
     T  A   T  A   Op  T  A   T  A   Op
    [3, 1] [2, 2] [3] [2, 1] [3, 1] [3]
    -> ["3,1", "2,2", "3", "2,1", "3,1", "3"]
    """
    if isinstance(batch[0], int):
        batch = [batch]  # type: ignore

    def chunk_5gram(chunk) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
        t1, a1, t2, a2, op = chunk
        return (t1, a1), (t2, a2), op

    def encode_single(tokens: List[int]) -> List[int]:
        chunks = [tokens[i : i + 5] for i in range(0, len(tokens), 5)]
        ngrams = [chunk_5gram(chunk) for chunk in chunks]
        flat = [item for sublist in ngrams for item in sublist]
        encoding = [vocab[item] for item in flat]
        return encoding

    encodings = [encode_single(tokens) for tokens in batch]  # type: ignore
    encodings = [[vocab[START], *enc, vocab[PAD]] for enc in encodings]

    encodings = pad_sequence(
        [torch.tensor(enc) for enc in encodings],
        batch_first=True,
        padding_value=vocab[PAD],
    )
    return encodings


def decode_batch(batch: torch.Tensor, rev_vocab) -> List[List[int]]:
    """
    Decode the batch of tokens back to the original flat encoding
    """
    if batch.ndim == 1:
        batch = batch.unsqueeze(0)

    def decode_single(tokens: torch.Tensor) -> List[int]:
        decode = tokens.tolist()
        decode = [rev_vocab[item] for item in decode]
        if PAD in decode:
            decode = decode[: decode.index(PAD)]

        flat = []
        for item in decode:
            if isinstance(item, tuple):
                flat.extend(list(item))
            else:
                flat.append(item)

        return flat

    def pad_short(tokens: List[int]) -> List[int]:
        """If the output sequence is too short (i.e. not a multiple of 5), we have to pad it."""
        if len(tokens) % 5 == 0:
            return tokens

        full_pad = [0, 2, 0, 2, 3]
        tokens = tokens + full_pad[len(tokens) % 5 :]
        return tokens

    # Cut out start
    batch = batch[:, 1:]
    decodings = [decode_single(tokens) for tokens in batch]
    decodings = [pad_short(dec) for dec in decodings]

    return decodings
