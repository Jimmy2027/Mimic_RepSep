# HK, 24.02.21
from scripts import tikz
from dataclasses import dataclass
import numpy as np


@dataclass
class Nodes:
    cl1: str = 'class 1'
    cl2: str = 'class 2'
    model: str = 'Model'
    cl1_: str = 'class 1'
    cl2_: str = 'class 2'
    lr:str = 'Latent Representation'

nodes = Nodes()
pic = tikz.Picture(
    'model/.style={rectangle, draw=red!60, fill=red!5, very thick, minimum height=20mm, minimum width=20mm},'
    'lr/.style={ellipse, draw=blue!60, fill=blue!5, very thick, minimum height=30mm,  minimum width=60mm},'
    't/.style={ellipse, draw=green!60, fill=green!5, very thick, minimum size=5mm},'
    'front/.style={ellipse, draw=orange!60, fill=orange!5, very thick, minimum size=5mm},'
)

pic.set_node(text=nodes.cl1, options='t', name='cl1')
pic.set_node(text=nodes.cl2, options='front, below of=cl1', name='cl2')
pic.set_node(text=nodes.model, options='model, right of=cl1, xshift=2cm, yshift=-0.5cm', name='model')
pic.set_node(text=nodes.lr, options='lr, right of=model, xshift=4cm', name='lr')
pic.set_node(text=nodes.cl1_, options='t, right of=lr, xshift=-1.8cm, yshift = -0.5cm', name='cl1_')
pic.set_node(text=nodes.cl2_, options='front, right of=lr, xshift=0.9cm, yshift = 0.3cm', name='cl2_')

pic.set_line('cl1', 'model', color='green!60')
pic.set_line('cl2', 'model', color='orange!60')
pic.set_line('model', 'cl1_', color='green!60')
pic.set_line('model', 'cl2_', color='orange!60')



output = pic.make()
# output = r'\resizebox{\textwidth}{!}{% ' + '\n' + pic.make() + '}'
print(output)
