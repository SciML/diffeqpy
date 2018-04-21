from julia import Julia

def setup():
    jul = Julia()
    jul.using("DiffEqPy")
    return jul
