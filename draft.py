import torch
import torch.nn as nn
import torch.optim as optim
import math
class SimpleTransformer(nn.Module):
    def __init__(self,d_model,vocab_size,n_head,num_layers):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.n_head=n_head
        self.num_layers=num_layers
        self.embed=nn.Embedding(vocab_size,d_model)
        self.pos=nn.Parameter(torch.zeros(1,vocab_size,d_model))  # 1*32*128
        self.transformer=nn.Transformer(d_model=d_model,nhead=n_head,num_encoder_layers=num_layers,num_decoder_layers=num_layers,batch_first=True)
        self.fc=nn.Linear(d_model,vocab_size)

    def forward(self,src,tgt):
        embed_src=self.embed(src)*math.sqrt(self.d_model)   # 32*10*128
        embed_src=embed_src+self.pos[:,:src.size(1),:]      # 

        embed_tgt=self.embed(tgt)*math.sqrt(self.d_model)
        embed_tgt=embed_tgt+self.pos[:,:tgt.size(1),:]

        tgt_mask=nn.Transformer.generate_square_subsequent_mask(embed_tgt.size(1)).to(tgt.device)

        out=self.transformer(src=embed_src,tgt=embed_tgt,tgt_mask=tgt_mask)
        out=self.fc(out)

        return out

model=SimpleTransformer()



class Transformer(nn.Module):
    def __init__(self,vocab_size,d_models):
        super().__init__()
        self.vocab_size=vocab_size
        self.d_models=d_models
        self.embed=nn.Embedding(vocab_size,d_models)
        self.pos_embed=nn.Parameter(torch.zeros(1,))