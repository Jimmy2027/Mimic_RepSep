# HK, 21.02.21

from scripts import tikz
from dataclasses import dataclass


@dataclass
class Nodes:
    input_L: str = 'L-Scan'
    input_F: str = 'F-Scan'
    input_text: str = 'Text'
    encoder: str = 'Encoder'
    lr: str = 'Latent Representation'
    decoder: str = 'Decoder'
    out_L: str = 'L-Scan'
    out_F: str = 'F-Scan'
    out_text: str = 'Text'


nodes = Nodes()
pic = tikz.Picture('modalities/.style={rectangle, draw=green!60, fill=green!5, very thick, minimum size=5mm},'
                   'model/.style={rectangle, draw=red!60, fill=red!5, very thick, minimum size=20mm},'
                   'lr/.style={ellipse, draw=blue!60, fill=blue!5, very thick, minimum size=15mm}')

pic.set_node(text=nodes.input_text, options='modalities', name='input_text')
pic.set_node(text=nodes.input_F, options='modalities, below of=input_text', name='input_F')
pic.set_node(text=nodes.input_L, options='modalities, below of=input_F', name='input_L')
pic.set_node(text=nodes.encoder, options='model, right of=input_F, xshift=2cm', name='encoder')
pic.set_node(text=nodes.lr, options='lr, right of=encoder, xshift=3cm', name='lr')
pic.set_node(text=nodes.decoder, options='model, right of=lr, xshift=3cm', name='decoder')
pic.set_node(text=nodes.out_F, options='modalities, right of=decoder, xshift=2cm', name='out_F')
pic.set_node(text=nodes.out_L, options='modalities, below of=out_F', name='out_L')
pic.set_node(text=nodes.out_text, options='modalities, above of=out_F', name='out_text')

pic.set_line('input_text', 'encoder')
pic.set_line('input_F', 'encoder')
pic.set_line('input_L', 'encoder')
pic.set_line('encoder', 'lr')
pic.set_line('lr', 'decoder')
pic.set_line('decoder', 'out_F')
pic.set_line('decoder', 'out_text')
pic.set_line('decoder', 'out_L')

output = r'\resizebox{\textwidth}{!}{% ' + '\n' + pic.make() + '}'
print(output)
