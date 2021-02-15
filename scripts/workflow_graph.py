# HK, 18.01.21
from scripts import tikz
from dataclasses import dataclass

IMG_SIZE = 128
IMG_ENCODER = 'ResNet'

@dataclass
class MMVAENodes:
    input_img: str = 'img dataset'
    input_text: str = 'text dataset'
    n1_text: str = 'text encoder'
    n1_img: str = IMG_ENCODER
    n2: str = 'lin. feature compressor'
    conc_latents: str = 'conc. latents for every mod in subset of powerset'
    poe: str = r'fuse with PoE:\\ $\mu_s = \sum_i ^{Subset}  \mu_i \cdot logvar^{-1}$\\$logvar_s = \sum_i^{Subset}\frac{1}{logvar_i}$'
    moe: str = 'fuse all subset latent with MoE'
    n3_text: str = r'lin. feature\\ generator text'
    n3_img: str = r'lin. feature\\ generator img'
    n4_text: str = 'text generator'
    n4_img: str = 'img generator'
    n5: str = 'likelihood'
    n6: str = 'log probability'
    loss: str = r'\textbf{Loss:}\\ $\sum_m^{\#mods} w_m \cdot log\_prob_m + \beta \cdot \beta_{content} \cdot joint\_divergence $'
    joint_divergence: str = r'\textbf{Joint Divergence:}\\ $\sum_{s \in subbset} kl\_div(\mu_s, logvar_s) \cdot \frac{1}{\#subsets}$\\ $kl\_div=\frac{1}{bs}\cdot -0.5 \cdot \sum_i^{bs} \sum_j^{class\_dim} 1 - exp(logvar)_{ij} -(\mu^2)_{ij} + logvar_ij$'
    batch: str = 'batch'


nodes = MMVAENodes()
pic = tikz.Picture('inference/.style={rectangle, draw=green!60, fill=green!5, very thick, minimum size=5mm},'
                   'rec/.style={rectangle, draw=red!60, fill=red!5, very thick, minimum size=5mm},'
                   'fuse_subsets/.style={rectangle, draw=orange!60, fill=orange!5, very thick, minimum size=5mm},'
                   'other/.style={rectangle, draw=gray!60, fill=gray!5, very thick, minimum size=5mm},'
                   'loss/.style={rectangle,rounded corners, draw=blue!60, fill=blue!5, very thick, minimum size=25mm}')

pic.set_node(text=nodes.input_img, options='inference', name='input_img')
pic.set_node(text=nodes.input_text, options='inference, right of=input_img, xshift=4cm', name='input_text')
pic.set_node(text=nodes.n1_text, options='inference, below of=input_text, yshift=-1cm', name='n1_text')
pic.set_node(text=nodes.n1_img, options='inference, below of=input_img, yshift=-1cm', name='n1_img')
pic.set_node(text=nodes.n2, options='inference, below of=n1_img, xshift=2cm,yshift=-1cm,text width=2cm,align=center',
             name='n2')
pic.set_node(text=nodes.conc_latents, options='fuse_subsets, right of=n2, text width=5cm,align=center, xshift=8cm',
             name='conc_latents')
pic.set_node(text=nodes.poe, options='fuse_subsets, below of=conc_latents, yshift=-1cm, align=center', name='poe')
pic.set_node(text=nodes.moe, options='fuse_subsets, below of=poe, yshift=-1cm', name='moe')
pic.set_node(text=nodes.n3_text, options='rec, left of=moe, xshift=-5cm, yshift=-1cm,text width=3cm,align=center',
             name='n3_text')
pic.set_node(text=nodes.n3_img, options='rec, left of=n3_text, xshift=-3cm,text width=3cm,align=center', name='n3_img')
pic.set_node(text=nodes.n4_img, options='rec, below of=n3_img, yshift=-1cm', name='n4_img')
pic.set_node(text=nodes.n4_text, options='rec, below of=n3_text, yshift=-1cm', name='n4_text')
pic.set_node(text=nodes.n5, options='rec, below of=n4_img, yshift=-1cm, xshift=2cm', name='n5')
pic.set_node(text=nodes.n6, options='rec, right of=n5, xshift=6cm', name='n6')
pic.set_node(text=nodes.loss, options='loss, below of=n4_text, yshift=-3cm, align=center', name='loss')
pic.set_node(text=nodes.joint_divergence, options='fuse_subsets, below of=n6, yshift=-4cm,xshift=-2cm, align=center',
             name='joint_divergence')
pic.set_node(text=nodes.batch, options='other, below of=moe, yshift=-1.5cm, align=center', name='batch')

pic.set_line('input_text', 'n1_text', label='[bs, sent len]', label_pos='west')
pic.set_line('n1_text', 'n2', label='[bs, 640, 1]', label_pos='west')
pic.set_line('input_img', 'n1_img', label=f'[bs, {IMG_SIZE}, {IMG_SIZE}]', label_pos='east')
pic.set_line('n1_img', 'n2', label='[bs, 320, 1]', label_pos='east')
pic.set_line('n2', 'conc_latents', label='mu\_content: [bs, class\_dim]', label_pos='south')
pic.set_line('n2', 'conc_latents', label='logvar\_content: [bs, class\_dim]', edge_options='bend right=10',
             label_pos='north')
# pic.set_line('n2', 'conc_latents', label='mu\_style: [bs, style\_dim]', label_pos='south')
# pic.set_line('n2', 'conc_latents', label='logvar\_style: [bs, style\_dim]')
pic.set_line('conc_latents', 'poe', label='[subset\_size, bs, class\_dim]', label_pos='west')
pic.set_line('poe', 'moe', label=r'[bs, class\_dim] $\forall$ subset', label_pos='west')
pic.set_line('moe', 'n3_text', label=r'[bs, class\_dim]', label_pos='south', edge_options='bend right=10')
pic.set_line('moe', 'n3_img', label=r'[bs, class\_dim]', label_pos='south', edge_options='bend right=20')
pic.set_line('n3_text', 'n4_text', label=r'[bs, 640]', label_pos='west')
pic.set_line('n3_img', 'n4_img', label=r'[bs, 320]', label_pos='east')
pic.set_line('n4_img', 'n5', label=f'[bs, 1, {IMG_SIZE}, {IMG_SIZE}]', label_pos='east')
pic.set_line('n4_text', 'n5', label=r'[bs, len sent, vocab size]', label_pos='west')
pic.set_line('n5', 'n6')
pic.set_line('batch', 'n6')
pic.set_line('n6', 'loss')
pic.set_line('poe', 'joint_divergence.east', label=r'[$\mu_s, logvar_s$]', edge_options='bend left=90',
             label_pos='east')
pic.set_line('joint_divergence', 'loss')

print(pic.make())
