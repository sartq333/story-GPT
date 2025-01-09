import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning

class Config:
    vocab_size = 50304
    n_epochs = 50
    batch_size = 36
    lr = 3e-4
    wd = 1e-6
    n_embed = 256
    num_blocks = 12
    num_heads = 12
    head_size = n_embed//num_heads
    context_len = 224
    attn_dropout_val = 0.2
    mha_dropout_val = 0.2
    ffn_dropout_val = 0.2

class CausalAttentionHead(nn.Module):
    def __init__(self, config):
        super(CausalAttentionHead, self).__init__()
        self.config = config

        self.query = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.key = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.attn_drop = nn.Dropout(config.attn_dropout_val)
        # mask for causal attention during training
        self.register_buffer("mask", torch.tril(torch.ones(config.context_len, config.context_len)))

    def forward(self, x):
        bs, context_len, embed_dim = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)
        attn_filter = torch.divide(torch.bmm(q, k.transpose(1, 2)), self.config.head_size)
        attn_filter = attn_filter.masked_fill(self.mask[:context_len, :context_len]==0, float("-inf"))
        attn_weights = F.softmax(attn_filter, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        output = torch.bmm(attn_weights, v)
        return output

class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadedAttention, self).__init__()
        self.config = config
        self.heads = nn.ModuleList(
            [CausalAttentionHead(config) for _ in range(config.num_heads)]
        )
        self.proj = nn.Linear(config.num_heads*config.head_size, config.n_embed)
        self.mha_drop = nn.Dropout(config.mha_dropout_val)

    def forward(self, x):
        mha_output = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.mha_drop(self.proj(mha_output))

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(FeedForwardNetwork, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(config.n_embed, config.n_embed*4),
            nn.GELU(),
            nn.Linear(config.n_embed*4, config.n_embed),
            nn.Dropout()
        )
    def forward(self, x):
        return self.ffn(x)

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.mha = MultiHeadedAttention(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ffn = FeedForwardNetwork(config)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = self.ln1(x+self.mha(x))
        x = self.ln2(x+self.ffn(x))
        return x

class GPT(lightning.LightningModule):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config
        self.save_hyperparameters()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.positional_embedding = nn.Embedding(config.context_len, config.n_embed)
        self.backbone = nn.Sequential(*[Block(config) for _ in range(config.num_blocks)])
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

    def forward(self, x):
        tok_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(torch.arange(x.shape[1], device=self.device))
        x = tok_emb+pos_emb
        x = self.backbone(x)
        logits = self.lm_head(x)
        return logits

    def get_loss(self, predictions, target):
        B, C, V = predictions.shape
        predictions = predictions.view(B*C, V)
        target = target.view(B*C)
        loss = F.cross_entropy(predictions, target)
        return loss

    def training_step(self, batch, batch_idx):
        text, target = batch
        text = text.long()
        target = target.long()
        logits = self(text)
        loss = self.get_loss(logits, target)

        self.log('loss', loss.item(), prog_bar=True)
        logs = {'loss': loss}

        return {"log": logs, "loss": loss}

    def training_end(self, outputs):
        avg_loss = torch.stack([x['log']['loss'] for x in outputs]).mean()
        logs = {"log": avg_loss}
        print(f"val_loss: {avg_loss}")
        return {"log": logs}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        return [opt], []

if __name__ == "__main__":
    config = Config() 
    gpt = GPT(config)