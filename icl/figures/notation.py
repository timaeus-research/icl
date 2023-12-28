LVAL = "L_\mathrm{val}"


def str_d_dt(s):
    return r"$\delta " + s + r"/\delta t$"


def str_d_dlogt(s):
    return r"$\delta " + s + r"/\delta\log t$"


def str_dlog_dlogt(s):
    return r"$\delta \log" + s + r"/\delta\log t$"


PYVAR_TO_MATHVAR = {
    "num_tasks": "M",
    "num_layers": "L",
    "num_heads": "H",
    "max_examples": "K",
    "lr": r"\eta",
    "elasticity": r"\gamma",
    "num_draws": r"n_\mathrm{draws}",
    "num_chains": r"n_\mathrm{chains}",
    "num_samples": r"n",
}
PYVAR_TO_SLUGVAR = {
    "num_tasks": "M",
    "num_layers": "L",
    "num_heads": "H",
    "max_examples": "K",
    "lr": "lr",
    "elasticity": "g",
    "num_draws": "ndraws",
    "num_chains": "nchains",
    "num_samples": "n",
    "epsilon": "eps",
    "temperature": "temp",
    "eval_method": "eval",
    "eval_loss_fn": "loss",
    "gamma": "gamma",
    "num_training_samples": "n",
    "batch_size": "m",
}


def pyvar_to_mathvar(name: str):
    return PYVAR_TO_MATHVAR.get(name.split(".")[-1].split("/")[-1], name)


def pyvar_to_slugvar(name: str):
    return PYVAR_TO_SLUGVAR.get(name.split(".")[-1].split("/")[-1])


def pyvar_dict_to_latex(d: dict):
    return "$" + ", ".join([f"{pyvar_to_mathvar(k)}={v}" for k, v in d.items() if v is not None and k in PYVAR_TO_MATHVAR]) + "$"


def pyvar_dict_to_slug(d: dict):
    return "_".join([f"{pyvar_to_slugvar(k)}{v}" for k, v in d.items() if v is not None and k in PYVAR_TO_SLUGVAR])
