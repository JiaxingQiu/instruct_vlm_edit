import os
import yaml
from types import SimpleNamespace


class AttrNS(SimpleNamespace):
    def to_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, AttrNS):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out


class NestedConfig(AttrNS):
    """Hydra-like nested config with back-compat attribute shims."""
    def __getattr__(self, name):
        # Back-compat shims for legacy flat access
        if name == "model_name":
            return getattr(self.model, "name", None)
        if name == "model_class":
            return getattr(self.model, "class_name", None)
        if name == "model_pt":
            return getattr(self.model, "pt", None)
        if name == "inner_params":
            return getattr(self.model, "inner_params", None)
        if name == "edit_lr":
            return getattr(self.editor, "edit_lr", None)
        if name == "task":
            return getattr(self.experiment, "task", None)
        if name == "dataset_name":
            return getattr(self.experiment, "dataset_name", None)
        return super().__getattr__(name)


def _load_yaml(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _to_ns(obj):
    if isinstance(obj, dict):
        return AttrNS(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


def configure_args(args, config_path=None):
    """Load config.yaml, merge editor preset, return nested config with shims. CLI overrides nested fields."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")

    # Base config
    cfg = _load_yaml(config_path)

    # Ensure expected top-level keys exist
    cfg.setdefault("model", {})
    cfg.setdefault("experiment", {})

    # Merge editor preset into nested editor block
    cli_editor = getattr(args, "editor", None)
    editor_cfg = {}
    if cli_editor:
        ep = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "editor", f"{cli_editor}.yaml")
        editor_cfg = _load_yaml(ep)
    # If config.yaml already set editor as string, prefer CLI string
    if isinstance(cfg.get("editor"), str):
        cfg["editor"] = {"_name": cfg["editor"]}
    # Overlay editor preset values (keep _name)
    if editor_cfg:
        name = editor_cfg.get("_name", cli_editor)
        merged = {k: v for k, v in cfg.get("editor", {}).items() if k != "_name"}
        for k, v in editor_cfg.items():
            if k == "_name":
                continue
            merged[k] = v
        cfg["editor"] = {"_name": name or cli_editor, **merged}
    elif "editor" not in cfg:
        # Default to CLI choice if provided
        if cli_editor:
            cfg["editor"] = {"_name": cli_editor}

    # CLI overrides for nested fields
    if getattr(args, "inner_params", None):
        cfg.setdefault("model", {})["inner_params"] = args.inner_params
    if getattr(args, "dataset_name", None):
        cfg.setdefault("experiment", {})["dataset_name"] = args.dataset_name
    if cli_editor and cfg.get("editor"):
        cfg["editor"]["_name"] = cli_editor

    # Move inner_params from top-level to model if present
    if "inner_params" in cfg and cfg["inner_params"]:
        cfg.setdefault("model", {})
        if "inner_params" not in cfg["model"] or not cfg["model"]["inner_params"]:
            cfg["model"]["inner_params"] = cfg["inner_params"]
        # Remove from top-level to avoid confusion
        del cfg["inner_params"]
    
    # Build nested namespace
    nested = {
        # pass through scalar top-levels
        "batch_size": cfg.get("batch_size", 1),
        "n_iter": cfg.get("n_iter", 100),
        "max_n_edits": cfg.get("max_n_edits", 5000),
        "seed": cfg.get("seed", 42),
        "device": cfg.get("device", "cuda"),
        "ckpt_dir": cfg.get("ckpt_dir", None),
        "dropout": cfg.get("dropout", None),
        # nested blocks
        "model": cfg.get("model", {}),
        "editor": cfg.get("editor", {"_name": cli_editor} if cli_editor else {}),
        "experiment": cfg.get("experiment", {}),
    }

    return NestedConfig(**_to_ns(nested).__dict__)


def build_args_from_yaml(config_path: str) -> NestedConfig:
    """Return a NestedConfig from a YAML file (for notebooks)."""
    cfg = _load_yaml(config_path)
    # Coerce editor string to nested
    if isinstance(cfg.get("editor"), str):
        cfg["editor"] = {"_name": cfg["editor"]}
    cfg.setdefault("model", {})
    cfg.setdefault("experiment", {})
    nested = {
        "batch_size": cfg.get("batch_size", 1),
        "n_iter": cfg.get("n_iter", 100),
        "max_n_edits": cfg.get("max_n_edits", 5000),
        "seed": cfg.get("seed", 42),
        "device": cfg.get("device", "cuda"),
        "ckpt_dir": cfg.get("ckpt_dir", None),
        "model": cfg.get("model", {}),
        "editor": cfg.get("editor", {}),
        "experiment": cfg.get("experiment", {}),
    }
    return NestedConfig(**_to_ns(nested).__dict__)
