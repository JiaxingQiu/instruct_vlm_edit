import torch
import higher
from higher.patch import monkeypatch as make_functional

from .utils import get_inner_params, brackets_to_periods, parent_module
import transformers
import torch.nn.functional as F


class GradientTransform(torch.nn.Module):
    """Transforms gradients for MEND"""
    def __init__(self, x_dim, delta_dim):
        super(GradientTransform, self).__init__()
        self.mlp1 = torch.nn.Linear(x_dim, x_dim)
        self.mlp2 = torch.nn.Linear(delta_dim, delta_dim)

    def forward(self, x, delta):
        if len(x.shape) == 3:
            x = x[:, -1, :]
            delta = delta[:, -1, :]
        return self.mlp1(x), self.mlp2(delta)


def hook_model(model, pnames):
    """Add hooks to model for MEND"""
    from .utils import hook_model as _hook_model
    _hook_model(model, pnames)


def get_shape(p, model):
    """Get shape for gradient transform"""
    return p.shape if isinstance(model, transformers.GPT2LMHeadModel) else (p.shape[1], p.shape[0])


class MEND(torch.nn.Module):
    """MEND: Model Editing Networks using Gradient Decomposition"""
    def __init__(self, config, model, tokenizer, device, mend=None):
        super(MEND, self).__init__()
        if mend is None:
            self.model = model.model if hasattr(model, 'model') else model
        else:
            self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
        self.pnames = [brackets_to_periods(config.inner_params[0])]
        
        hook_model(self.model, self.pnames)
        
        if not isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True
            
        if mend is None:
            self.mend = torch.nn.ModuleDict({})
            for n, p in get_inner_params(self.model.named_parameters(), self.pnames):
                shape = get_shape(p, self.model)
                if transpose:
                    self.mend[n.replace(".", "#")] = GradientTransform(shape[0], shape[1]).to(device)
                else:
                    self.mend[n.replace(".", "#")] = GradientTransform(shape[1], shape[0]).to(device)
        else:
            self.mend = mend
            
    def outer_parameters(self):
        return list(self.mend.parameters())
    
    def forward(self, **kwargs):
        return self.model(**kwargs)

    def get_model_loss(self, model, logits, batch):
        if hasattr(model, "get_loss"):
            return model.get_loss(logits, batch)
        if hasattr(model, "model") and hasattr(model.model, "get_loss"):
            return model.model.get_loss(logits, batch)
        return None

    def edit(self, config, tokens, batch_history):
        opt = torch.optim.Adam(self.outer_parameters(), lr=config.edit_lr)
        n_iter = config.n_iter

        for i in range(n_iter):
            edited_model = self.edit_step(tokens)
            if i == 0:
                with torch.no_grad():
                    for p_new, p_old in zip(edited_model.parameters(), self.model.parameters()):
                        p_old.copy_(p_new)
            
            self.loss = edited_model(**tokens).loss
            self.loss.backward()
            opt.step()
            opt.zero_grad()

    def edit_step(self, batch):
        outputs = self.model(**batch)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = outputs.loss if hasattr(outputs, "loss") else None
        
        if loss is None:
            if "labels" in batch:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    batch["labels"].view(-1), 
                    ignore_index=-100
                )
            else:
                return self.model
        
        loss.backward()

        transformed_factors = {
            n: self.mend[n.replace(".", "#")](p.__x__, p.__delta__)
            for n, p in get_inner_params(self.model.named_parameters(), self.pnames)
        }

        mean_grads = {
            n: torch.matmul(delta.view(-1, 1), x.view(1, -1))
            for n, (x, delta) in transformed_factors.items()
        }
        
        self.model.zero_grad()
        
        edited_model = self.model
        if not isinstance(edited_model, higher.patch._MonkeyPatchBase):
            edited_model = make_functional(edited_model, device=self.device)

        new_params = []
        updates = mean_grads
        for n, p in edited_model.named_parameters():
            if n in self.pnames:
                new_params.append(p + updates[n].T)
            else:
                new_params.append(p)

        loss.detach()
        edited_model.update_params(new_params)
        return MEND(self.config, edited_model, self.tokenizer, self.device, self.mend)

