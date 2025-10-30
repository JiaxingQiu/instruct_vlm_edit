import torch


def get_inner_params(named_parameters, inner_names):
    """Get parameters by name"""
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names if n in param_dict]


def param_subset(named_parameters, inner_names):
    """Get subset of parameters"""
    param_dict = dict(named_parameters)
    return [param_dict[n] for n in inner_names if n in param_dict]


def parent_module(model, pname):
    """Get parent module for a parameter name"""
    components = pname.split('.')
    parent = model
    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component}")
    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]}")
    return parent


def brackets_to_periods(name):
    """Convert brackets to periods in parameter names"""
    return name.replace("[", ".").replace("]", "")


def linear_backward_hook(mod, grad_in, grad_out):
    """Hook for capturing gradients in MEND"""
    if not hasattr(mod, "weight"):
        return
    if hasattr(mod.weight, "__x__"):
        assert len(grad_out) == 1
        mod.weight.__delta__ = grad_out[0].detach()


def linear_forward_hook(mod, activations, output):
    """Hook for capturing activations in MEND"""
    assert len(activations) == 1
    mod.weight.__x__ = activations[0].detach()


def hook_model(model, pnames):
    """Add forward and backward hooks to model for MEND"""
    handles = []
    for pname in pnames:
        parent = parent_module(model, pname)
        handles.append(parent.register_forward_hook(linear_forward_hook))
        handles.append(parent.register_full_backward_hook(linear_backward_hook))
    model.handles = handles


