from dezero import *
import numpy as np
import os, subprocess
def get_dot_graph(y:Variable, verbose=False):
    dot_file = "digraph g{\n"
    dot_file += f"{id(y)} [label=\"{y.name}\", color=orange, style=filled]\n"
    dot_file += f"{id(y.creator)} -> {id(y)}\n"
    funcs = [y.creator]
    while funcs:
        f = funcs.pop()
        if type(f).__name__ != "Pow":
            f_name = type(f).__name__
        else:
            f_name = f"Pow({f.c})"
        dot_file += f"{id(f)} [label=\"{f_name}\", color=lightblue, style=filled, shape=box]\n"    
        
        for x in f.inputs:
            if x.name is None:
                if x.creator is None:
                    x.name = x.data
                else:
                    x.name = ""

            dot_file += f"{id(x)} -> {id(f)}\n"
            dot_file += f"{id(x)} [label=\"{x.name}\", color=orange, style=filled]\n" if not verbose \
                        else f"{id(x)} [label=\"{x.name}:{x.shape} {x.dtype}\", color=orange, style=filled]\n"

            if x.creator is not None and x.creator not in funcs:
                dot_file += f"{id(x.creator)} -> {id(x)}\n"
                funcs.append(x.creator)
    dot_file += "}\n"
    return dot_file

def plot_dot_graph(output, to_file:str, verbose=False):
    dot_file_txt = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    os.makedirs(tmp_dir, exist_ok=True)
    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with open(graph_path, 'w') as f:
        f.write(dot_file_txt)
    
    extension = os.path.splitext(to_file)[1].replace(".", "")
    cmd = f'dot {graph_path} -T {extension} -o {to_file}'
    subprocess.run(cmd, shell=True)