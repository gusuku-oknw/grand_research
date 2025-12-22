# -*- coding: utf-8 -*-
from pathlib import Path
p = Path('thesis.tex')
text = p.read_text(encoding='utf-8-sig')
line = 'シェア数 $r$ に応じて、$r<k_1$ はノイズ、$k_1 \\le r < k_2$ は pHash 符号一致ダミー、$r \\ge k_2$ は原本を復元する。\n'
fig = """\\begin{figure}[t]\n  \\centering\n  \\includegraphics[width=0.8\\linewidth]{output/fig_scene.png}\n  \\caption{三段階 SIS の概念図（$k_1$:検索のみ，$k_2$:完全復元）}\n  \\label{fig:three-stage}\n\\end{figure}\n"""
if line not in text:
    raise SystemExit('target line not found')
text = text.replace(line, line + fig, 1)
p.write_text(text, encoding='utf-8-sig')
print('inserted fig')
