import collections

import torch
import einops
from jaxtyping import Array, Float, Int

from stirling.retnet import RetNet as RetNetStirling


def make_scale_matrix(nseq : int,
                      scale : Float[Array, "nheads"]) -> Float[Array, "nheads nseq nseq"]:
    n = nseq
    expoent = (torch.arange(n)[..., None] - torch.arange(n)).to(scale) #[n, n]
    posmask = torch.log((expoent >= 0).to(scale)) #[n, n]    
    scale = scale[..., None, None]
    logmask = expoent*torch.log(scale) + posmask #[nheads, n, n]
    mask = torch.exp(logmask)
    return mask


def make_angles(x : Float[Array, "nbatch nseq nhead dhead"],
                offset : int = 0,
                conjugate : bool = False):
    #x : (nbatch, nseq, nhead, dhead)
    _, nseq, _, dhead = x.shape
    expoent = (torch.arange(0, dhead, 2)/dhead) #[dhead/2]
    angle = 10**(-4*expoent) #[dhead/2]
    if conjugate:
        angle = -angle
    nangle = torch.einsum('i, j -> ij', torch.arange(offset, nseq+offset), angle).to(x) #[nseq, dhead/2]
    return nangle


class XPos2(torch.nn.Module):
    """
        The module implementing rotation for the complex plane
    """
    def __init__(self, nheads : int = 1):
        super().__init__()
        self.nheads = nheads

    def forward(self, x : (Float[Array, "nbatch nseq nhead dhead"]),
                offset = 0,
                conjugate : bool = False) -> Float[Array, "nbatch nseq nhead dhead"]:
        #x : shape [nbatch, nseq, nhead, dhead]
        nbatch, nseq, nheads, dhead = x.shape
        assert(dhead%2 == 0)
        dhead_c = dhead//2
        x_c = einops.rearrange(x, 'batch seq head (dhead_c two) -> batch seq head dhead_c two',
                               dhead_c=dhead_c, two=2) #[nbatch, nseq, nhead, dhead_c, 2] #Complexification
        nangle = make_angles(x, offset=offset, conjugate=conjugate) #[nseq, dhead_c] #Makes both theta and n
        #The next two lines actually make the rotation in the complex plane
        sin, cos = torch.sin(nangle).unsqueeze(1), torch.cos(nangle).unsqueeze(1) #[nseq, 1, dhead_c]
        x_c_rotated = torch.stack([x_c[..., 0]*cos - x_c[..., 1]*sin,
                                   x_c[..., 0]*sin + x_c[..., 1]*cos],
                                  axis=-1) #[nbatch, nseq, nhead, dhead, 2]
        x_rotated = x_c_rotated.flatten(start_dim=-2) #[nbatch, nseq, nhead, dhead] #Realfication
        return x_rotated

    def forward_no_seq_dim(self, x : Float[Array, "nbatch nhead dhead"],
                           offset : int = 0,
                           conjugate : bool = False) -> Float[Array, "nbatch nhead dhead"]:
        x = x.unsqueeze(1)
        x = self.forward(x, offset, conjugate)
        x = x.squeeze(1)
        return x


class EmbedGroupNorm(torch.nn.Module):
    """
        A wrapper for torch.nn.GroupNorm, to ensure it group-normalizes only the last dimension,
        unlike the original torch.nn.GroupNorm
    """
    def __init__(self, dmodel : int, nheads : int):
        super().__init__()
        self.gn = torch.nn.GroupNorm(nheads, dmodel)

    def forward(self, x : Float[Array, "batch seq dim"]) -> Float[Array, "batch seq dim"]:
        b, n, d = x.shape
        x = einops.rearrange(x, 'b n d -> (b n) d', b=b, n=n) #Rearrange for group norme
        x = self.gn(x)
        x = einops.rearrange(x, '(b n) d -> b n d', b=b, n=n) #Derearrange
        return x

    def forward_no_seq_dim(self, x : Float[Array, "batch dim"]) -> Float[Array, "batch dim"]:
        return self.gn(x)


class MSRBlock(torch.nn.Module):
    """
        This is the retentive block substituting the attention block.
    """
    def __init__(self, dmodel : int, nheads : int, has_wg : bool = False):
        super().__init__()
        self.dmodel = dmodel
        self.nheads = nheads
        self.dhead = dmodel//nheads
        assert(self.dhead*self.nheads == self.dmodel)
        assert(self.dhead%2 == 0)
        self.dummy = torch.nn.Parameter(torch.ones([1]))
        self.xpos = XPos2(self.nheads)
        self.gn = EmbedGroupNorm(self.dmodel, self.nheads)
        self.act = torch.nn.SiLU()
        self.has_wg = has_wg
        # Apply Xavier Initialization
        self.WQ = torch.nn.Parameter(torch.empty([nheads, self.dhead, dmodel]))
        self.WK = torch.nn.Parameter(torch.empty([nheads, self.dhead, dmodel]))
        self.WV = torch.nn.Parameter(torch.empty([nheads, self.dhead, dmodel]))
        self.WO = torch.nn.Parameter(torch.empty([dmodel, dmodel]))
        weights = [self.WQ, self.WK, self.WV, self.WO]
        if self.has_wg:
            self.WG = torch.nn.Parameter(torch.empty([dmodel, dmodel]))
            weights.append(self.WG)
        for W in weights:
            torch.nn.init.xavier_normal_(W)
        scale = 1.0 - (2.0)**(-5-torch.arange(self.nheads)) #[nhead]
        self.register_buffer("scale", scale)
        self.register_buffer("state", self.default_state)
        self.offset = 1

    def recurrent_retention_step(self, x : Float[Array, "batch dim"],
                                 old_state : Float[Array, "batch head dim dim"],
                                 step : int) -> tuple[Float[Array, "batch dim"],
                                                      Float[Array, "batch head dim dim"],
                                                      int]:
        assert(len(x.shape) == 2)
        nbatch, dmodel = x.shape
        Q = torch.einsum('bd, hud -> bhu', x, self.WQ) #[nbatch, nheads, dhead] #The key tensor
        K = torch.einsum('bd, hud -> bhu', x, self.WK) #[nbatch, nheads, dhead] #The query tensor
        Q = self.xpos.forward_no_seq_dim(Q, offset=step) #[nbatch, nheads, dhead]
        K = self.xpos.forward_no_seq_dim(K, offset=step) #[nbatch, nheads, dhead]
        V = torch.einsum('bd, hud -> bhu', x, self.WV) #[nbatch, nheads, dhead]
        KV = torch.einsum('nhi, nhj -> nhij', K, V) #[nbatch, nhead, dhead, dhead]
        scale = self.scale[:, None, None] #[nhead, 1, 1]
        S = scale*old_state + KV #[nbatch, nhead, dhead, dhead]
        R = torch.einsum('bhj, bhjd -> bhd', Q, S)
        R = einops.rearrange(R, 'nbatch nheads dhead -> nbatch (nheads dhead)', nheads=self.nheads, dhead=self.dhead)
        R = self.gn.forward_no_seq_dim(R)
        increment = 1
        return R, S, increment

    def parallel_retention(self, x : Float[Array, "batch seq dim"]) -> Float[Array, "batch seq dim"]:
        #dmodel = (dhead nheads)
        nseq = x.shape[-2]
        Q = torch.einsum('bsd, hud -> bshu', x, self.WQ) #[nbatch, nseq, nheads, dhead]
        K = torch.einsum('bsd, hud -> bshu', x, self.WK) #[nbatch, nseq, nheads, dhead]
        Q = self.xpos(Q) #[nbatch, nseq, nheads, dhead]
        K = self.xpos(K) #[nbatch, nseq, nheads, dhead]
        QK = torch.einsum('bihd, bjhd -> bhij', Q, K) #(nbatch, nheads, nseq, nseq)
        D = make_scale_matrix(nseq, self.scale) #(nheads, nseq, nseq)
        QKD = QK*D #(nbatch, nheads, nseq, nseq)
        V = torch.einsum('bsd, hud -> bshu', x, self.WV) #[nbatch, nseq, nheads, dhead]
        R = torch.einsum('bhij, bjhd -> bihd', QKD, V) #[batch, nheads, nseq, dhead]
        R = einops.rearrange(R, 'nbatch nseq nheads dhead -> nbatch nseq (nheads dhead)', nheads=self.nheads, dhead=self.dhead)
        #R : [nbatch, nseq, dmodel]
        R = self.gn(R) #[nbatch, nseq, dmodel]
        return R

    def forward(self, x : Float[Array, "batch seq dim"]) -> Float[Array, "batch seq dim"]:
        # return R
        R = self.parallel_retention(x)
        if self.has_wg:
            G = self.act(x@self.WG) #[nbatch, nseq, model]
            R = R*G
        y = R@self.WO #[nbatch, nseq, model]
        y = R@self.WO
        return y

    def serialized_forward(self, x : Float[Array, "batch dim"],
                           reset_state : bool = False) -> Float[Array, "batch seq dim"]:
        if reset_state:
            self.reset_state()
        R, state, increment = self.recurrent_retention_step(x, self.state, self.offset)
        self.state = state
        self.offset += 1
        #R : [nbatch, dmodel]
        if self.has_wg:
            G = self.act(x@self.WG) #[nbatch, dmodel]
            R = R*G
        y = R@self.WO
        y = y.unsqueeze(1) #[nbatch, 1, model]
        return y

    def chunk_forward(self, x, reset_state=False):
        raise NotImplementedError

    def serialized_forward_seq(self, x : Float[Array, "batch seq dim"],
                               reset_state : bool = False) -> Float[Array, "batch seq dim"]:
        if reset_state:
            self.reset_state()
        nseq = x.shape[1]
        y = torch.zeros_like(x)
        for ind in range(nseq):
            xi = x[:, ind, :]
            y[:, ind, :] = self.serialized_forward(xi)[:, 0, :]
        return y

    def reset_state(self):
        self.state = self.default_state
        self.offset = 0

    @property
    def default_state(self):
        return torch.zeros([1, self.nheads, self.dhead, self.dhead]).to(self.device)

    @property
    def device(self):
        return self.dummy.device


class FFNBlock(torch.nn.Module):
    def __init__(self, dmodel : int, nhidden : int):
        super().__init__()
        act = torch.nn.GELU()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dmodel, nhidden),
                                       act,
                                       torch.nn.Linear(nhidden, dmodel))

    def forward(self, x : Float[Array, "batch seq dim"]) -> Float[Array, "batch seq dim"]:
        return self.mlp(x)

class RetBlock(torch.nn.Module):
    """
    Implements an alternative version of GPT-2 encoder,
    where the attention block is substituted by retentive block
    """
    def __init__(self, dmodel : int, nheads : int, nhidden : int, pdrop : float = 0.0, has_wg : bool = False):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(dmodel)
        self.ln2 = torch.nn.LayerNorm(dmodel)
        self.msr = MSRBlock(dmodel, nheads, has_wg=has_wg) #Retentive Block
        self.ffn = FFNBlock(dmodel, nhidden) #FFN
        self.dropout = torch.nn.Dropout(pdrop)

    def forward(self, x : Float[Array, "batch seq dim"]) -> Float[Array, "batch seq dim"]:
        # y = self.msr(x) + x
        y = self.dropout(self.msr(self.ln1(x))) + x
        y = self.dropout(self.ffn(self.ln2(y))) + y
        return y


class RetentionEncoder(torch.nn.Module):
    """
    Implements an alternative version of GPT-2 encoder,
    where the attention blocks are substituted by retentive blocks
    """

    def __init__(self, nlayers : int, nheads : int, dmodel : int, nhidden : int = 128,
                 pdrop : float = 0.0, has_wg : bool = False):
        super().__init__()
        self.nlayers = nlayers
        self.layers = torch.nn.ModuleList(RetBlock(dmodel, nheads, nhidden, pdrop, has_wg=has_wg) for _ in range(nlayers))
        self.ln = torch.nn.LayerNorm(dmodel)

    def forward(self, values : Float[Array, "batch seq dim"]) -> Float[Array, "batch seq dim"]:
        for layer in self.layers:
            values = layer(values)
        values = self.ln(values)
        return values


class GPTEncoder(torch.nn.Module):
    def __init__(self, nlayers : int, nheads : int,
                 dmodel : int, nhidden : int = 128,
                 pdrop : float = 0.0):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(dmodel,
                                                         nheads,
                                                         nhidden,
                                                         dropout=pdrop,
                                                         batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer,
                                                   nlayers)
    
    def forward(self, x):
        mask = self.generate_square_subsequent_mask(x.shape[1]).to(x.device)
        return self.encoder(x, mask, is_causal=False)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class GPTR(torch.nn.Module):
    """
    Implements an alternative version of GPT-2, where the attention blocks are substituted by retentive blocks
    """
    def __init__(self, nvocab : int,
                 nctx : int | None = None,
                 dembed : int=64,
                 nlayers : int=4,
                 nheads : int=4,
                 nhidden : int | None = None,
                 pdrop : float = 0.0,
                 has_wg : bool = False,
                 decoder_mode : str = "default"):
        super().__init__()
        assert (decoder_mode in ["default", "stirling", "transformer"])
        self.nvocab = nvocab
        self.nctx = nctx
        self.embed = torch.nn.Embedding(nvocab, dembed, padding_idx=0)
        if self.nctx is not None:
            self.pos = torch.nn.Parameter(torch.zeros([nctx, dembed]))
            torch.nn.init.xavier_uniform_(self.pos)
        nhidden = nhidden if nhidden is not None else 4*dembed
        if decoder_mode == "stirling":
            self.decoder = RetNetStirling(nlayers, dembed, nhidden, nheads)
        elif decoder_mode == "transformer":
            self.decoder = GPTEncoder(nlayers, nheads, dembed, nhidden, pdrop)
        elif decoder_mode == "default":
            self.decoder = RetentionEncoder(nlayers, nheads, dembed, nhidden, pdrop, has_wg=has_wg)
        self.projection = torch.nn.Linear(dembed, nvocab)
        self.dropout = torch.nn.Dropout(pdrop)

    def forward(self, tokens : Int[Array, "batch tokens"],
                apply_softmax : bool = False,
                apply_positional_embedding : bool = False) -> Float[Array, "batch tokens vocab"]:
        #tokens : (..., ntokens)
        x = self.decode(tokens)
        x = self.projection(x) #(..., ntokens, dmodel)
        if apply_softmax:
            x = torch.softmax(x, dim=-1) #(..., ntokens, nvocab)
        else:
            x = torch.nn.functional.log_softmax(x, dim=-1) #(..., ntokens, nvocab)
        return x

    def decode(self, tokens : Int[Array, "batch tokens"],
               apply_positional_embedding : bool = True) -> Int[Array, "batch tokens"]:
        #tokens : (..., ntokens)
        d = tokens.shape[-1]
        xe = self.embed(tokens)
        if apply_positional_embedding:
            if self.nctx is None:
                raise ValueError
            d = tokens.shape[-1]
            pos = self.pos[:d, :]
            xe += pos
        xe = self.dropout(xe)
        x = self.decoder(xe)
        return x


class GPTRConfig(object):
    def __init__(self, vocab_size,
                 context_window=None,
                 embedding_dim=64,
                 nlayers=6,
                 nheads=4,
                 nhidden=None,
                 nclasses=None,
                 pdrop=0.0):
        self.vocab_size = vocab_size
        self.context_window = context_window
        self.embedding_dim = embedding_dim
        self.nlayers = nlayers
        self.nheads = nheads
        self.nhidden = nhidden
        if self.nhidden is None:
            self.nhidden = 4*self.embedding_dim
        self.nclasses = nclasses
        self.pdrop = pdrop

    def from_model_config(model_config):
        vocab_size = model_config.vocab_size
        context_window = model_config.max_position_embeddings
        embedding_dim = model_config.hidden_size
        nlayers = model_config.num_hidden_layers
        nheads = model_config.num_attention_heads
        nhidden = model_config.intermediate_size
        nclasses = model_config.num_labels
        return GPTRConfig(vocab_size,
                          context_window,
                          embedding_dim,
                          nlayers,
                          nheads,
                          nhidden,
                          nclasses)


class GPTRAutoregressive(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = GPTR(config.vocab_size,
                          config.context_window,
                          config.embedding_dim,
                          config.nlayers,
                          config.nheads,
                          config.nhidden)

    def forward(self, x):
        return self.model(x)


class GPTRClassifier(torch.nn.Module):
    def __init__(self, config, has_wg=True, decoder_mode="default"):
        super().__init__()
        assert config.nclasses is not None
        self.model = GPTR(config.vocab_size,
                          config.context_window,
                          config.embedding_dim,
                          config.nlayers,
                          config.nheads,
                          config.nhidden,
                          has_wg=has_wg,
                          decoder_mode=decoder_mode)
        self.classifier = torch.nn.Linear(config.embedding_dim,
                                          config.nclasses)

    def forward(self, input_ids, lengths):
        x = input_ids
        index_seq = lengths-1
        index_batch = torch.arange(x.shape[0])
        x = self.model.decode(x)
        x = self.classifier(x)
        x = x[index_batch, index_seq, :]
        output_cls = collections.namedtuple("output", ["logits"])
        res = output_cls(logits=x)
        return res
