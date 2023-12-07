LVAL = "L_\mathrm{val}"


def wrap_latex(s):
    return "$" + s + "$"

def add_d_dlogt(s):
    return wrap_latex(r"\delta " + s + r"/\delta\log t")