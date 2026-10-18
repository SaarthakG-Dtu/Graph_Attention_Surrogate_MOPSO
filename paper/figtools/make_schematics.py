"""Copy the draft's TikZ schematics, update them for the two-output surrogate and the three applications, raise every
font below 8 pt to 8 pt, and compile to ../figures_v2/ (sources in ../schematics/tex are not changed).

  python -I make_schematics.py
"""
import re, shutil, subprocess, pathlib
HERE = pathlib.Path(__file__).parent
SRC = pathlib.Path(r"E:\SUTD_BENCHMARK_NOVEL_GNN\paper\schematics\tex")
TEX = HERE.parent / "schematics_v2" / "tex"; TEX.mkdir(parents=True, exist_ok=True)     # never the sources in SRC
FIG = HERE.parent / "figures_v2"


FIG_PT = 8      # one type size in every figure (the authors' choice; IEEE asks for one consistent size)


def min8(s):
    """Every \\fontsize{a}{b} becomes \\fontsize{FIG_PT}{b'}, the leading scaled with the size (at least FIG_PT + 1)."""
    def f(m):
        a, b = float(m.group(1)), float(m.group(2))
        return r"\fontsize{%g}{%.1f}" % (FIG_PT, max(b * FIG_PT / a, FIG_PT + 1))
    return re.sub(r"\\fontsize\{([\d.]+)\}\{([\d.]+)\}", f, s)


SANS = (r"\usepackage[T1]{fontenc}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}"
        r"\usepackage{newtxsf}")    # figures in Helvetica with sans-serif math (IEEE graphics fonts)


def edit(src, dst, reps):
    s = (SRC / src).read_text(encoding="utf-8")
    reps = [(r"\usepackage{newtxtext,newtxmath}", SANS)] + list(reps)
    for o, n in reps:
        assert s.count(o) == 1, (src, o[:70])
        s = s.replace(o, n)
    (TEX / dst).write_text(min8(s), encoding="utf-8")


# ---------------------------------------------------------------- architecture: two-output passive head
edit("fig_architecture.tex", "fig_architecture.tex", [
 (r"\node[ttl,text=headC!80!black,anchor=north west] at (11.3,7.6) {passive polar head and output};",
  r"\node[ttl,text=headC!80!black,anchor=north west] at (11.3,7.6) {passive two-output head and output};"),
 (r"\node[blk=headC,minimum width=1.05cm,minimum height=1.25cm,inner sep=1.5pt] (head) at (12.0,5.95) {Linear\\$(a_k,b_k)$\\[2pt]$\hat r_k=\sigma(a_k)$\\$\times[\cos b_k$\\$+\,j\sin b_k]$};",
  r"\node[blk=headC,minimum width=1.05cm,minimum height=1.25cm,inner sep=1.5pt] (head) at (12.05,5.95) {Linear\\$a,c,b,b'$\\[2pt]$\rho=\sigma(a)$\\$\psi=\tfrac{\pi}{2}\sigma(c)$};"),
 (r"  \addplot[black!55,dashed,line width=0.5pt] table[x=re,y=im] {../data/spectrum.dat};" "\n"
  r"  \addplot[mesh,point meta=\thisrow{f},line width=1.1pt] table[x=rep,y=imp] {../data/spectrum.dat};",
  r"  \addplot[black!55,dashed,line width=0.5pt] table[x=re,y=im] {../data/spectrum.dat};" "\n"
  r"  \addplot[black!55,dashed,line width=0.5pt] table[x=reY,y=imY] {../data/spectrum.dat};" "\n"
  r"  \addplot[pgaC,line width=1.0pt] table[x=rep,y=imp] {../data/spectrum.dat};" "\n"
  r"  \addplot[bbC,line width=1.0pt] table[x=repY,y=impY] {../data/spectrum.dat};"),
 (r"{$|\hat r|{=}1$};", r"{$\rho{=}1$};"),
 (r"\node[sub] at (13.85,4.55) {$\hat r_{xx}(f)$, unit disk};",
  r"\node[sub] at (13.85,4.55) {\textcolor{pgaC}{$\hat r_{xx}$}, \textcolor{bbC}{$\hat r_{yx}$}; $|\hat r_{xx}|^2{+}|\hat r_{yx}|^2{=}\rho^2$};"),
 # second spectrum panel: |r_yx| instead of the phase of r_xx
 (r"  xmin=2,xmax=12,ymin=-180,ymax=180,xtick={2,7,12},ytick={-180,180},",
  r"  xmin=2,xmax=12,ymin=0,ymax=1.05,xtick={2,7,12},ytick={0,1},"),
 (r"  \addplot[black!60,dashed,line width=0.55pt,unbounded coords=jump] table[x=f,y=phR] {../data/spectrum.dat};" "\n"
  r"  \addplot[pgaC,line width=0.85pt] table[x=f,y=phRp] {../data/spectrum.dat};" "\n"
  r"  \node[font=\fontsize{7.7}{8.0}\selectfont,text=ink,anchor=south west] at (axis cs:2.3,-170) {$\angle r_{xx}$};",
  r"  \addplot[black!60,dashed,line width=0.55pt] table[x=f,y=absY] {../data/spectrum.dat};" "\n"
  r"  \addplot[bbC,line width=0.85pt] table[x=f,y=absYp] {../data/spectrum.dat};" "\n"
  r"  \node[font=\fontsize{7.7}{8.0}\selectfont,text=ink,anchor=north west] at (axis cs:2.3,1.0) {$|r_{yx}|$};"),
])

# ---------------------------------------------------------------- method: three applications, FDTD check
edit("fig_pipeline.tex", "fig_method.tex", [
 (r"$\leftrightarrow$ latent $\mathbf z\in\mathbb R^{150}$", r"$\leftrightarrow$ latent $\mathbf z\in\mathbb R^{200}$"),
 (r"  0.15\,ms per layout\\(CST: about 10\,min)};", r"  0.13\,ms per layout\\(CST: about 10\,min)};"),
 (r"\node[ttl,text=headC!85!black,anchor=north west] at (9.3,12.45) {(b) surrogate RDN108000 (trained once, Sec.\ III)};",
  r"\node[ttl,text=headC!85!black,anchor=north west] at (9.3,12.45) {(b) surrogate (trained once, Sec.\ IV)};"),
 (r"\layerbar{13.35}{10.62}{0.72}{headC}{polar head}", r"\layerbar{13.35}{10.62}{0.72}{headC}{2 outputs}"),
 (r"\node[sub,anchor=west,align=left] at (14.3,10.75) {co-pol reflection\\$\hat r_{xx}(f)$ at 101 frequencies\\[2pt]",
  r"\node[sub,anchor=west,align=left] at (14.3,10.75) {$\hat r_{xx}(f)$, $\hat r_{yx}(f)$\\at 101 frequencies\\[2pt]"),
 (r"\stephead{0.25}{8.7}{1}{swarm $\mathbf z_i\in[-3,3]^{150}$}", r"\stephead{0.25}{8.7}{1}{swarm: pixels or $\mathbf z_i$}"),
 (r"\stephead{4.6}{8.7}{2}{decode with $D_\theta$}", r"\stephead{4.6}{8.7}{2}{decode ($D_\theta$ or $>0.5$)}"),
 # step 3: magnitudes and axial ratio of the best LP-to-CP design
 (r"\draw[step=attC] (9.15,6.25) rectangle (13.6,8.7);", r"\draw[step=attC] (9.15,6.25) rectangle (13.3,8.7);"),
 (r"\draw[step=attC] (14.1,6.25) rectangle (17.85,8.7);", r"\draw[step=attC] (13.8,6.25) rectangle (17.85,8.7);"),
 (r"\stephead{9.15}{8.7}{3}{surrogate predicts $\hat r_{xx}(f)$}", r"\stephead{9.15}{8.7}{3}{surrogate: \textcolor{pgaC}{$|\hat r_{xx}|$}, \textcolor{bbC}{$|\hat r_{yx}|$}}"),
 (r"  xmin=2,xmax=12,ymin=-180,ymax=180,xtick={2,4,6,8,10,12},xticklabels={},ytick={-90,90},",
  r"  xmin=6,xmax=12,ymin=0,ymax=1.05,xtick={6,8,10,12},xticklabels={},ytick={0,1},"),
 (r"  \fill[attC!18] (axis cs:9.7,-180) rectangle (axis cs:10.8,180);" "\n"
  r"  \draw[muted,dashed,line width=0.4pt] (axis cs:2,90) -- (axis cs:12,90);" "\n"
  r"  \draw[muted,dashed,line width=0.4pt] (axis cs:2,-90) -- (axis cs:12,-90);" "\n"
  r"  \addplot[bbC,line width=0.9pt] table[x=f,y=ph] {../data/design.dat};" "\n"
  r"  \node[font=\fontsize{7.4}{8}\selectfont,text=bbC,anchor=west] at (axis cs:2.3,-20) {phase};" "\n"
  r"  \node[font=\fontsize{7.4}{8}\selectfont\bfseries,text=attC!60!black] at (axis cs:10.25,-120) {$B$};",
  r"  \fill[attC!18] (axis cs:9.0,0) rectangle (axis cs:10.95,1.05);" "\n"
  r"  \addplot[pgaC,line width=0.9pt] table[x=f,y=rxx] {../data/design.dat};" "\n"
  r"  \addplot[bbC,line width=0.9pt] table[x=f,y=ryx] {../data/design.dat};"),
 (r"  xmin=2,xmax=12,ymin=-12,ymax=1,xtick={2,7,12},xticklabels={},ytick={-10,0},",
  r"  xmin=6,xmax=12,ymin=0,ymax=6,xtick={6,8,10,12},ytick={0,3},"),
 (r"  \fill[attC!18] (axis cs:9.7,-12) rectangle (axis cs:10.8,1);" "\n"
  r"  \addplot[pgaC,line width=0.9pt] table[x=f,y=db] {../data/design.dat};" "\n"
  r"  \node[font=\fontsize{7.4}{8}\selectfont,text=pgaC,anchor=south west] at (axis cs:2.2,-11.5) {$|r_{xx}|$ dB};",
  r"  \fill[attC!18] (axis cs:9.0,0) rectangle (axis cs:10.95,6);" "\n"
  r"  \draw[muted,dashed,line width=0.4pt] (axis cs:6,3) -- (axis cs:12,3);" "\n"
  r"  \addplot[ink,line width=0.9pt] table[x=f,y=ar] {../data/design.dat};" "\n"
  r"  \node[font=\fontsize{8}{9}\selectfont,text=ink,anchor=west] at (axis cs:6.05,4.4) {AR};" "\n"
  r"  \node[font=\fontsize{8}{9}\selectfont\bfseries,text=attC!60!black] at (axis cs:9.97,4.6) {$B$};"),
 # step 4: objectives of the three devices
 (r"\stephead{14.1}{8.7}{4}{three objectives}", r"\stephead{13.8}{8.7}{4}{three objectives}"),
 (r"\node[lbl,anchor=north,align=center,font=\fontsize{8.2}{11.4}\selectfont] at (15.975,8.05) {%" "\n"
  r"  $B$: in-phase bandwidth (\%) $\uparrow$\\" "\n"
  r"  $L$: lowest $|r_{xx}|$ in band (dB) $\uparrow$\\" "\n"
  r"  $D$: fragile features $\downarrow$\\[2pt]" "\n"
  r"  {\color{muted}\fontsize{7.6}{9}\selectfont band around 10\,GHz}};",
  r"\node[lbl,anchor=north,align=left,font=\fontsize{8.2}{9.9}\selectfont] at (15.83,8.2) {%" "\n"
  r"  CP: AR band $B\uparrow$, $\overline{\mathrm{AR}}\downarrow$\\" "\n"
  r"  rotator: PCR band $B\uparrow$, $\overline{|r_{yx}|^2}\uparrow$\\" "\n"
  r"  coding: $-10$\,dB band $B\uparrow$, $\bar R\uparrow$\\" "\n"
  r"  all: $D\downarrow$ (corner contacts\\\ \ and isolated pixels)};"),
 (r"\draw[flow] (13.6,7.45) -- (14.1,7.45);", r"\draw[flow] (13.3,7.45) -- (13.8,7.45);"),
 # step 5: archive in (B, AR, D) of the LP-to-CP converter
 (r"  xmin=0,xmax=20,ymin=-20,ymax=0,zmin=0,zmax=60,xtick={0,10,20},ytick={-20,-10,0},ztick={0,30,60},",
  r"  xmin=0,xmax=22,ymin=0,ymax=3,zmin=0,zmax=50,xtick={10,20},ytick={1,2},ztick={0,25,50},"),
 (r"  xlabel={$B$ (\%)},ylabel={$L$ (dB)},zlabel={$D$},", r"  xlabel={$B$ (\%)},ylabel={AR (dB)},zlabel={$D$},"),
 (r"  \addplot3[only marks,mark=*,mark size=0.45pt,black!22] table[x=B,y=L,z=D] {../data/cloud.dat};" "\n"
  r"  \addplot3[only marks,mark=square*,mark size=1.0pt,draw=pgaC!70!black,fill=pgaC!80] table[x=B,y=L,z=D] {../data/front.dat};" "\n"
  r"  \addplot3[only marks,mark=*,mark size=1.3pt,draw=attC!60!black,fill=attC] table[x=B,y=L,z=D] {../data/archive.dat};",
  r"  \addplot3[only marks,mark=*,mark size=0.45pt,black!22] table[x=B,y=AR,z=D] {../data/cloud.dat};" "\n"
  r"  \addplot3[only marks,mark=square*,mark size=1.0pt,draw=pgaC!70!black,fill=pgaC!80] table[x=B,y=AR,z=D] {../data/front.dat};" "\n"
  r"  \addplot3[only marks,mark=*,mark size=1.3pt,draw=attC!60!black,fill=attC] table[x=B,y=AR,z=D] {../data/archive.dat};"),
 (r"\stephead{11.4}{5.75}{5}{non-dominated archive (50 designs)}", r"\stephead{11.4}{5.75}{5}{archive: Pareto front (50 designs)}"),
 (r"  $\mathbf v_i\leftarrow w\mathbf v_i+c_1(\mathbf p_i-\mathbf z_i)+c_2(\mathbf g-\mathbf z_i)$, \ $\mathbf z_i\leftarrow\mathbf z_i+\mathbf v_i$\\",
  r"  $\mathbf v_i\leftarrow w\mathbf v_i+c_1\mathbf r_1\!\odot\!(\mathbf p_i-\mathbf z_i)+c_2\mathbf r_2\!\odot\!(\mathbf g-\mathbf z_i)$, \ $\mathbf z_i\leftarrow\mathbf z_i+\mathbf v_i$\\"),
 (r"  mutation: half the swarm, step $10\,\%\to2\,\%$ of the range};",
  r"  velocity clipped to 0.042 (pixels), 0.2 (VAE); mutation:\\half the swarm, step $10\,\%\to2\,\%$ of the range};"),
 (r"\node[lbl,anchor=center,align=center,font=\fontsize{8.4}{11.2}\selectfont] at (6.95,3.85) {%",
  r"\node[lbl,anchor=center,align=center,font=\fontsize{8.4}{11.2}\selectfont] at (6.95,4.05) {%"),
 # (d) full-wave check with FDTD
 (r"\node[sub,anchor=north,align=center] at (15.125,1.95) {after the final generation:\\6 designs of the long VAE search\\" "\n"
  r"  and 3 of the pixel search closest\\to the ideal point};",
  r"\node[sub,anchor=north,align=center] at (15.125,1.95) {6 designs closest to the ideal\\point, plus designs predicted\\" "\n"
  r"  to meet the target (at most 12)};"),
 (r"\node[lbl,anchor=north,font=\fontsize{8.8}{10}\selectfont\bfseries] at (9.1,2.4) {CST simulation};",
  r"\node[lbl,anchor=north,font=\fontsize{8.8}{10}\selectfont\bfseries] at (9.1,2.4) {FDTD and CST};"),
 (r"\node[sub,anchor=center,align=center] at (10.35,1.1) {Floquet ports\\2--12\,GHz\\$t=1.0$\,mm, $\varepsilon_r=2.65$};",
  r"\node[sub,anchor=center,align=center] at (10.35,1.1) {FDTD: $h\to0$,\\copper loss,\\supercells; CST};"),
 (r"\node[lbl,anchor=north,font=\fontsize{8.8}{10}\selectfont\bfseries] at (3.025,2.4) {prediction vs.\ CST};",
  r"\node[lbl,anchor=north,font=\fontsize{8.8}{10}\selectfont\bfseries] at (3.025,2.4) {prediction vs.\ full wave};"),
 (r"\node[sub,anchor=north,align=center] at (3.025,1.95) {errors of $|r_{xx}|$ and phase;\\" "\n"
  r"  $B$, $L$, $D$ predicted and simulated\\(Table~VII, Fig.~8)};",
  r"\node[sub,anchor=north,align=center] at (3.025,1.95) {objectives and spectra,\\" "\n"
  r"  predicted and simulated;\\Monte Carlo of tolerances};"),
])

# ---------------------------------------------------------------- unit cell: both reflected components
edit("fig_unitcell.tex", "fig_unitcell.tex", [
 (r"\node[lbl,text=bbC!85!black,anchor=south] at (6.6,6.9,9.3) {reflected, $r_{xx}$};",
  r"\node[lbl,text=bbC!85!black,anchor=south] at (6.6,6.9,9.3) {$r_{xx}$};" "\n"
  r"\wave{8.6}{8.6}{2.6}{8.6}{pgaC}{0}{0.9}" "\n"
  r"\draw[->,pgaC!80!black,line width=1.0pt] (8.6,8.6,6.8) -- (8.6,8.6,9.3);" "\n"
  r"\node[lbl,text=pgaC!85!black,anchor=south] at (8.6,8.6,9.3) {$r_{yx}$};"),
])


def resize(name, reps):
    p = TEX / f"{name}.tex"; t = p.read_text(encoding="utf-8")
    for o, n in reps:
        assert t.count(o) == 1, (name, o); t = t.replace(o, n)
    p.write_text(t, encoding="utf-8")


# printed size: text width 18.19 cm (figure*), column width 8.89 cm; no scaling in LaTeX
resize("fig_architecture", [("[border=2pt]", "[border=1pt]"), ("[x=1cm,y=1.12cm]", "[x=0.975cm,y=1.12cm]"),
                            ("at={(15.45cm,6.664cm)}", "at={(15.05cm,6.664cm)}"), ("at={(15.45cm,5.432cm)}", "at={(15.05cm,5.432cm)}"),
                            ("at={(12.72cm,5.174cm)}", "at={(12.45cm,5.174cm)}"),
                            (r"\node[sub,anchor=south east] at (18.1,7.05)", r"\node[sub,anchor=south east] at (17.75,7.05)"),
                            (r"; $|\hat r_{xx}|^2{+}|\hat r_{yx}|^2{=}\rho^2$};", r" in the unit disk};")])
resize("fig_method", [("[border=2pt]", "[border=1pt]"), ("[x=1cm,y=1cm]", "[x=0.985cm,y=1cm]"),
                      ("at={(9.95cm,7.2cm)},anchor=south west,width=3.4cm", "at={(9.75cm,7.2cm)},anchor=south west,width=2.95cm"),
                      ("at={(9.95cm,6.55cm)},anchor=south west,width=3.4cm", "at={(9.75cm,6.55cm)},anchor=south west,width=2.95cm"),
                      ("xlabel style={sloped like x axis,yshift=5pt},ylabel style={sloped like y axis,yshift=5pt}",
                       "xlabel style={xshift=2pt,yshift=2pt},ylabel style={xshift=-2pt,yshift=2pt}")])
resize("fig_unitcell", [("[border=2pt]", "[border=1pt]"),
                        ("[x={(0.36cm,-0.145cm)},y={(0.25cm,0.18cm)},z={(0cm,0.36cm)}]",
                         "[x={(0.25cm,-0.10cm)},y={(0.175cm,0.126cm)},z={(0cm,0.25cm)}]"),
                        (r"\node[sub,anchor=north west] at (-3.9,-0.6,-2.6) {thicknesses\\[-1pt]not to scale};", ""),
                        (r"node[sub,anchor=north east] {$p=10$\,mm}", r"node[sub,anchor=north,yshift=-1pt] {$p=10$\,mm}"),
                        (r"at (12.6,0,\zt/2+0.55) {substrate", r"at (13.6,0,\zt/2+0.4) {substrate"),
                        (r"at (13.4,6.5,\zc+0.6) {$16\times16$ copper pixels", r"at (13.4,6.5,\zc+3.4) {$16\times16$ copper pixels"),
                        (r"(8.2,6.8,\zc) -- (13.3,6.6,\zc+0.6)", r"(8.2,6.8,\zc) -- (13.3,6.6,\zc+3.4)"),
                        (r"at (12.3,1.5,\zg-1.35) {copper ground", r"at (12.8,1.5,\zg-2.6) {copper ground"),
                        (r"(10,3,\zg/2) -- (12.2,1.6,\zg-1.25)", r"(10,3,\zg/2) -- (12.7,1.6,\zg-2.5)")])


# 9 pt text: move or shorten the labels that now collide (the shortcut note goes to the caption)
resize("fig_architecture", [
    (r"\node[sub,anchor=north west,align=left,text width=1.5cm] at (15.55,11.0) {$\mathcal S$: identity;\\[-1pt]$1{\times}1$ conv\\[-1pt]+ BN at a\\[-1pt]stage start\\[-1pt](stride 2)};", ""),
    (r"\node[lbl,anchor=north] at (2.86,8.68) {stem\\[-2pt]{\color{muted}\fontsize{8}{9.0}\selectfont $3{\times}3$ conv}};",
     r"\node[lbl,anchor=north] at (2.86,8.68) {stem};"),
    (r"\node[sub] at (13.85,4.55) {\textcolor{pgaC}{$\hat r_{xx}$}, \textcolor{bbC}{$\hat r_{yx}$} in the unit disk};",
     r"\node[sub] at (13.6,4.55) {\textcolor{pgaC}{$\hat r_{xx}$}, \textcolor{bbC}{$\hat r_{yx}$}};"),
    (r"node[lbl,anchor=north,yshift=1pt] {$\mathbf x_k$}", r"node[lbl,anchor=east,xshift=-1pt] {$\mathbf x_k$}"),
    # sans-serif text is wider
    (r"at (11.3,7.6) {passive two-output head and output};", r"at (11.3,7.6) {passive two-output head};"),
    (r"\node[lbl,anchor=south] at (0.62,2.25) {$F\in\mathbb{R}^{C\times H\times W}$};",
     r"\node[lbl,anchor=south] at (0.66,2.4) {$F\in\mathbb{R}^{C\times H\times W}$};"),
    (r"{$F'$\\[-1pt]{\color{muted}learned gate $\gamma$}};", r"{$F'$\\[-1pt]{\color{muted}gate $\gamma$}};"),
    (r"\node[sub,anchor=north] at (15.35,0.5) {$\mathbf h_k", r"\node[sub,anchor=north] at (15.65,0.5) {$\mathbf h_k"),
    (r"\node[sub,anchor=north] at (15.35,0.15) {$\tilde{\mathbf h}_k", r"\node[sub,anchor=north] at (15.65,0.15) {$\tilde{\mathbf h}_k")])
resize("fig_method", [
    (r"  CP: AR band $B\uparrow$, $\overline{\mathrm{AR}}\downarrow$\\", r"  CP: $B\uparrow$, $\overline{\mathrm{AR}}\downarrow$\\"),
    (r"  rotator: PCR band $B\uparrow$, $\overline{|r_{yx}|^2}\uparrow$\\", r"  rotator: $B\uparrow$, $\overline{|r_{yx}|^2}\uparrow$\\"),
    (r"  coding: $-10$\,dB band $B\uparrow$, $\bar R\uparrow$\\", r"  coding: $B\uparrow$, $\bar R\uparrow$\\"),
    (r"+c_2\mathbf r_2\!\odot\!(\mathbf g-\mathbf z_i)$, \ $\mathbf z_i\leftarrow\mathbf z_i+\mathbf v_i$\\",
     r"+c_2\mathbf r_2\!\odot\!(\mathbf g-\mathbf z_i)$\\$\mathbf z_i\leftarrow\mathbf z_i+\mathbf v_i$\\")])
resize("fig_unitcell", [(r"node[sub,anchor=north,yshift=-1pt] {$p=10$\,mm}", r"node[sub,anchor=north,yshift=-5pt] {$p=10$\,mm}"),
                        ("\\end{tikzpicture}", "\\node[anchor=north west,inner sep=0pt,font=\\fontsize{8}{9}\\selectfont] at "
                                               "(current bounding box.north west) {(a)};\n\\end{tikzpicture}")])


def compile_tex(name):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"], cwd=TEX,
                       capture_output=True, text=True, errors="replace")
    pdf = TEX / f"{name}.pdf"
    if r.returncode or not pdf.exists():
        raise RuntimeError(r.stdout[-2500:])
    shutil.copy(pdf, FIG / f"{name}.pdf")
    import fitz
    page = fitz.open(pdf)[0]
    print(f"{name}: {page.rect.width / 72 * 2.54:.2f} x {page.rect.height / 72 * 2.54:.2f} cm")


for n in ("fig_architecture", "fig_method", "fig_unitcell"):
    compile_tex(n)
