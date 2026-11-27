# opinf_rom_heat_eqn

A self-contained introduction to reduced-order models (ROMs) and Operator Inference (OpInf),
built around one toy problem: the heat equation on 7 grid points, reduced to 2 dimensions.

## Start here
Open `note/opinf_rom_heat_eqn.pdf`. It contains:
1. Background: full-order model, basis/projection/lifting, Galerkin (intrusive) ROM, POD via SVD
2. Intrusive vs. non-intrusive, definition of Operator Inference, when OpInf = intrusive ROM
3. The toy example solved completely by hand (8 steps), checked against code
4. The main script explained block by block, plus every numpy function on a tiny example
5. Example 2: closure error and Peherstorfer's re-projection
6. Take-aways and the link to sparse regression (SINDy / MCP / FBS)

## Folder layout
```
note/
  opinf_rom_heat_eqn.tex   LaTeX source
  opinf_rom_heat_eqn.pdf   compiled note (22 pages)
  figures/                 5 figures (made by code/make_figures.py)
  generated/               script outputs and mini examples, \input by the note
code/
  opinf_heat_demo.py       main script: FOM -> POD -> OpInf -> compare -> predict
  hand_check.py            checks every number of the hand calculation
  closure_reprojection.py  Example 2 (data outside the POD subspace)
  make_figures.py          writes note/figures/*.pdf
  mini_examples.py         writes note/generated/mini_examples.tex
```

## Reproduce everything
Requires Python 3 with numpy and matplotlib, and a LaTeX installation.
```
cd code
python opinf_heat_demo.py
python hand_check.py
python closure_reprojection.py
python make_figures.py
python mini_examples.py
python opinf_heat_demo.py      > ../note/generated/opinf_heat_demo.out
python hand_check.py           > ../note/generated/hand_check.out
python closure_reprojection.py > ../note/generated/closure_reprojection.out
cd ../note
pdflatex opinf_rom_heat_eqn.tex && pdflatex opinf_rom_heat_eqn.tex
```

## Main references
- B. Peherstorfer, K. Willcox, CMAME 306 (2016) 196-215 (original OpInf paper)
- B. Kramer, B. Peherstorfer, K. Willcox, Annu. Rev. Fluid Mech. 56 (2024) 521-548 (review)
- B. Peherstorfer, SIAM J. Sci. Comput. 42(5) (2020) (re-projection)
- Python package and tutorials: https://operator-inference.github.io/opinf/
