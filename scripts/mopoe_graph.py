from scripts import tikz
from dataclasses import dataclass
import numpy as np


@dataclass
class Nodes:
    input_L: str = 'L-Scan'
    input_F: str = 'F-Scan'
    input_text: str = 'Text'
    encoder: str = 'Encoder'
    poe: str = r'\textbf{PoE}\\ \vspace{\baselineskip}\\ $\prod \limits _{\textbf{x}_j \in \xsubset}q_{\phi_j}(\textbf{z}|\textbf{x}_j)$'
    moe: str = r'\textbf{MoE}\\ \vspace{\baselineskip}\\ $\frac{1}{2^3} \sum \limits _{\textbf{x}_k \in \mathbb{X}} \tilde{q}_{\phi} (\textbf{z}|\mathbb{X}_k)$'
    z: str = r'joint\\ posterior'
    points: str = r'\ldots'


nodes = Nodes()
pic = tikz.Picture(
    'model/.style={rectangle, draw=red!60, fill=red!5, very thick, minimum height=40mm, minimum width=20mm},'
    # 'lr/.style={ellipse, draw=blue!60, fill=blue!5, very thick, minimum size=15mm},'
    't/.style={rectangle, draw=green!60, fill=green!5, very thick, minimum size=5mm},'
    'lat/.style={rectangle, draw=blue!60, fill=blue!5, very thick, minimum size=5mm},'
    'front/.style={rectangle, draw=orange!60, fill=orange!5, very thick, minimum size=5mm},'
    'lr/.style={circle, draw=gray!60, fill=gray!5, very thick, minimum size=5mm},'
)

pic.set_node(text=nodes.input_text, options='t', name='input_text')
pic.set_node(text=nodes.input_F, options='front, below of=input_text', name='input_F')
pic.set_node(text=nodes.input_L, options='lat, below of=input_F', name='input_L')
pic.set_node(text=nodes.encoder, options='model, right of=input_F, xshift=1cm', name='encoder')
pic.set_node(text=nodes.poe, options='model, right of=encoder, xshift=3cm, align=center', name='poe')
pic.set_node(text=nodes.points, options=' right of=poe, xshift=1cm', name='points')
pic.set_node(text=nodes.moe, options='model, right of=points, xshift=1.5cm, align=center', name='moe')
pic.set_node(text=nodes.z, options='lr,right of=moe, xshift=3cm, align=center', name='z')
pic.set_line('input_text', 'encoder')
pic.set_line('input_F', 'encoder')
pic.set_line('input_L', 'encoder')

mod_color = {0: 'green', 1: 'orange', 2: 'blue'}
mod = 0
for idx, i in enumerate(np.linspace(-1.8, 1.8, 6)):
    if idx % 2 == 0:
        pic.set_line(f'[yshift=-{i}cm]encoder.east', f'[yshift=-{i}cm]poe.west', label_pos='south',
                     label=r'\textcolor{' + mod_color[mod] + '}{$\mu' + f'_{mod}$}}')
    else:
        pic.set_line(f'[yshift=-{i}cm]encoder.east', f'[yshift=-{i}cm]poe.west', label_pos='south',
                     label=r'\textcolor{' + mod_color[mod] + '}{$\sigma' + f'_{mod}$}}')
        mod += 1

mods = {0: 0, 2: '7'}
mod = 0
for idx, i in enumerate(np.linspace(-1.8, 1.8, 6)):
    if idx % 2 == 0:
        if mod != 1:
            pic.set_line(f'[yshift=-{i}cm]poe.east', f'[yshift=-{i}cm]moe.west', label_pos='south',
                         label=r'$\mu ^\prime' + f'_{mods[mod]}$')
    else:
        if mod != 1:
            pic.set_line(f'[yshift=-{i}cm]poe.east', f'[yshift=-{i}cm]moe.west', label_pos='south',
                         label=r'$\sigma ^\prime' + f'_{mods[mod]}$')
        mod += 1

pic.set_line('[yshift=-1cm]moe.east', '[yshift=-5mm]z.west', label=r'$\mu$', label_pos='south')
pic.set_line('[yshift=1cm]moe.east', '[yshift=5mm]z.west', label=r'$\sigma$', label_pos='south')


output = r'\resizebox{\textwidth}{!}{% ' + '\n' + pic.make() + '}'
print(output)
