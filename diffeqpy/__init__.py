from julia import Julia

def setup():
    jul = Julia(debug=True)
    jul.using("DiffEqPy")
    return jul
