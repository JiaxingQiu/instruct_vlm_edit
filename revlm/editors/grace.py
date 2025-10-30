import torch
import torch.nn.functional as F
from .utils import parent_module, brackets_to_periods
import transformers


def mmd(query, key):
    """Maximum Mean Discrepancy distance"""
    kdist = torch.exp(-torch.cdist(key, key)).mean(-1).mean(-1)
    qdist = torch.exp(-torch.cdist(query, query)).mean(-1).mean(-1)
    kqdist = torch.exp(-torch.cdist(query, key)).mean(-1).mean(-1)
    return kdist + qdist - kqdist


def cos(query, key, eps=1e-8):
    """Cosine distance"""
    if len(key.shape) < 2:
        key = key.view(1, -1)
    query_n, key_n = query.norm(dim=1)[:, None], key.norm(dim=1)[:, None]
    query_norm = query / torch.clamp(query_n, min=eps)
    key_norm = key / torch.clamp(key_n, min=eps)
    sim_mt = torch.mm(key_norm, query_norm.T)
    return 1-sim_mt


def euc(query, key):
    """Euclidean distance"""
    if len(key.shape) < 2:
        key = key.view(1, -1)
    return torch.cdist(key, query, p=2)


def pairwise_dist(query, keys, dist_fn):
    """Compute distance from query to all keys"""
    dists = []
    if dist_fn == "mmd":
        d_fn = mmd
    elif dist_fn == "cos":
        d_fn = cos
    elif dist_fn == "euc":
        d_fn = euc
    else:
        raise ValueError(f"Distance name {dist_fn} does not exist")

    for i in range(len(keys)):
        dists.append(d_fn(query, keys[i]).view(-1, 1))
    return torch.stack(dists).view(-1, len(query))


class GRACE(torch.nn.Module):
    """GRACE: General Retrieval Adaptors for Continual Editing"""
    def __init__(self, config, model):
        super(GRACE, self).__init__()
        self.config = config
        self.log_dict = {}
        self.model = model.model if hasattr(model, 'model') else model
        self.tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        
        layer = config.inner_params[0]
        self.device = config.device

        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer
        
        for n, p in self.model.named_parameters():
            p.requires_grad = False
        
        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        layer_name = self.layer.rsplit(".", 1)[-1]
        original_layer = getattr(edit_module, layer_name)
        setattr(edit_module, layer_name, GRACEAdaptor(config, original_layer, transpose=transpose).to(self.device))
        
    def __call__(self, **kwargs):
        if self.config.task == "hallucination":
            key_id = (kwargs.get("labels", torch.tensor([])) == -100).sum() - 1
            setattr(eval(f"self.model.{self.layer}"), "key_id", key_id)
        return self.model(**kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
        
    def edit(self, config, tokens, batch_history):
        if hasattr(config, 'task') and config.task == "hallucination":
            key_id = (tokens.get("labels", torch.tensor([])) == -100).sum() - 1
            setattr(eval(f"self.model.{self.layer}"), "key_id", key_id)
        
        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "edit_label", tokens.get("labels", None))
                
        self.losses = []
        n_iter = config.n_iter
        edit_lr = config.edit_lr
        
        for i in range(n_iter):
            setattr(eval(f"self.model.{self.layer}"), "iter", i)
            outputs = self.model(**tokens)
            
            if i == 0:
                optimizer = torch.optim.Adam(self.model.parameters(), edit_lr)
            
            loss = outputs.loss if hasattr(outputs, "loss") else None
            if loss is None:
                break
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.losses.append(loss.detach().cpu().numpy())
        
        self.loss = loss if 'loss' in locals() else None
        setattr(eval(f"self.model.{self.layer}"), "training", False)
        
        # Log info (only if attributes exist)
        layer_obj = eval(f"self.model.{self.layer}")
        if hasattr(layer_obj, "chosen_key"):
            self.log_dict["chosen_key"] = getattr(layer_obj, "chosen_key")
        if hasattr(layer_obj, "keys"):
            self.log_dict["nkeys"] = len(getattr(layer_obj, "keys"))


class GRACEAdaptor(torch.nn.Module):
    def __init__(self, config, layer, transpose):
        super(GRACEAdaptor, self).__init__()
        self.layer = layer
        self.init_epsilon = getattr(config, 'eps', 0.1)
        self.dist_fn = getattr(config, 'dist_fn', 'cos')
        self.replacement = getattr(config, 'replacement', 'replace_all')
        self.device = layer.weight.device
        self.config = config
        self.num_pert = getattr(config, 'num_pert', 10)
        self.key_id = -1
    
        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]
        self.training = False

    def add_key(self, new_key, new_value):
        keys = torch.vstack([self.keys, new_key.detach()])
        values = torch.nn.Parameter(torch.vstack([self.values, new_value]), requires_grad=True)
        new_epsilon = torch.tensor(self.init_epsilon, device=self.device).view(1)
        epsilons = torch.vstack([self.epsilons, new_epsilon])
        key_labels = self.key_labels + [self.edit_label]
        return keys, values, epsilons, key_labels

    def init_key_value(self, query, value):
        key = query.detach()
        value = value
        epsilon = torch.tensor(self.init_epsilon, device=self.device, requires_grad=False).view(1)
        key_label = [self.edit_label]
        return key, value, epsilon, key_label

    def label_match(self, edit_label, key_label):
        if isinstance(edit_label, torch.Tensor) and isinstance(key_label, torch.Tensor):
            return edit_label.float().mean() == key_label.float().mean()
        return edit_label == key_label

    def split_epsilons_in_half(self, nearest_key, smallest_distance):
        self.epsilons[nearest_key] = (smallest_distance / 2) - 1e-5
        self.epsilons[-1] = smallest_distance / 2
    
    def forward(self, *args):
        layer_out = self.layer(*args)
        
        if not hasattr(self, 'keys') or not self.training:
            # First pass or inference - initialize or use existing
            if not hasattr(self, 'keys'):
                # Initialize on first forward pass during training
                query = args[0][:, self.key_id, :]
                value_out = layer_out[:, self.key_id, :] if len(layer_out.shape) == 3 else layer_out
                
                key, value, epsilon, key_label = self.init_key_value(query, value_out)
                self.keys = key
                self.values = torch.nn.Parameter(value, requires_grad=True)
                self.epsilons = epsilon
                self.key_labels = key_label
                
                if self.training:
                    return layer_out
                else:
                    return value_out.unsqueeze(0) if len(value_out.shape) == 1 else value_out
        
        query = args[0][:, self.key_id, :]
        
        # Find nearest key
        distances = pairwise_dist(query.unsqueeze(0), [self.keys[i] for i in range(len(self.keys))], self.dist_fn)
        nearest_key = torch.argmin(distances, dim=0).item()
        smallest_distance = distances[nearest_key].item()
        
        # Check if we should use existing key or create new one
        if self.replacement == "replace_all" or smallest_distance > self.epsilons[nearest_key]:
            # Add new key
            value_out = layer_out[:, self.key_id, :] if len(layer_out.shape) == 3 else layer_out
            key, value, epsilon, key_label = self.init_key_value(query, value_out)
            self.keys, self.values, self.epsilons, self.key_labels = self.add_key(key, value)
            self.chosen_key = len(self.keys) - 1
            
            if smallest_distance <= self.epsilons[nearest_key]:
                self.split_epsilons_in_half(nearest_key, smallest_distance)
        else:
            # Use nearest key
            self.chosen_key = nearest_key
        
        # Return value for chosen key
        chosen_value = self.values[self.chosen_key]
        if len(layer_out.shape) == 3:
            layer_out[:, self.key_id, :] = chosen_value
            return layer_out
        else:
            return chosen_value.unsqueeze(0) if len(chosen_value.shape) == 1 else chosen_value

