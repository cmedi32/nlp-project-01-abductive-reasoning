"""Architecture diagrams of the two final Project 1 models, end to end with tensor shapes, white background.

    python3 presentation/project_1/architecture_diagrams.py

Writes a PNG (200 dpi, 3200 x 1800) and an SVG per model next to this file, plus one slide that zooms into the
TF-IDF weighting both models share. Shapes and parameter counts follow the configs of the two final notebooks:
  notebooks/project_1/static/final_presentable_notebook/glove_tfidf_mlp.ipynb
  notebooks/project_1/rnn/final_presentable_notebook/glove_lstm_bidirectional_improved.ipynb
The example is train[0] of amarfurt/art-nlp-hs26, tokenized with pipeline/preprocessing/nltk_alnum_lower.py.
The idf values come from pipeline/features/tfidf.py (fit_vectorizer + idf_weights on the training split).
"""
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath

OUT_DIR = Path(__file__).resolve().parent

# cloudtec colors
INK = "#030C26"  # dark blue: text and borders
WHITE = "#FFFFFF"
GREEN = "#02FFA1"  # trainable weights
BLUE = "#01E7FF"  # observations O1, O2
PURPLE = "#A064FF"  # hypothesis H1 / story A
YELLOW = "#F9F871"  # hypothesis H2 / story B
GRAY_1, GRAY_2, GRAY_4 = "#F2F3F5", "#E4E5EA", "#BFC3CD"
GRAY_7, GRAY_9, GRAY_11 = "#767E94", "#646C86", "#313B56"

STYLES = {
    "trainable": dict(fc=GREEN, ec=INK, ls="-"),
    "fixed": dict(fc=GRAY_2, ec=GRAY_7, ls="-"),  # not trained: rules, frozen GloVe, fixed idf
    "no_weights": dict(fc=WHITE, ec=GRAY_7, ls=(0, (3, 2))),
    "input": dict(fc=WHITE, ec=GRAY_4, ls="-"),
    "container": dict(fc="none", ec=GRAY_4, ls="-"),
}

LANE_X0, LANE_PITCH, LANE_W = 34, 31, 29  # four lanes O1, O2, H1, H2 next to the 31-unit label margin
LANES = ["O1", "O2", "H1", "H2"]
LANE_COLORS = {"O1": BLUE, "O2": BLUE, "H1": PURPLE, "H2": YELLOW}
SENTENCES = {  # train[0], label 2: H2 is the more plausible hypothesis
    "O1": "A baby panda was lost in the woods.",
    "O2": "He was reunited with his parents, who also lived at the zoo.",
    "H1": "He was never lost.",
    "H2": "All the people from the zoo looked for him.",
}
TOKENS = {
    "O1": "a baby panda was lost in the woods",
    "O2": "he was reunited with his parents who also lived at the zoo",
    "H1": "he was never lost",
    "H2": "all the people from the zoo looked for him",
}

# tf-idf of H1, fitted on the training split: token, idf, df (training sentences that contain the token)
TRAIN_SENTENCES = 67_204  # 4 x 16,801, every sentence is one document
TFIDF_EXAMPLE = [("he", 3.1971, 7_467), ("was", 2.4655, 15_521), ("never", 5.0980, 1_115), ("lost", 5.8295, 536)]
IDF_CONTEXT = [("the", 2.0556, "most common", "center"), ("panda", 9.5506, "in 12 sentences", "center"),  # last: label alignment
               ("rarest", 11.4224, "once in train", "right"), ("unseen", 12.1155, "not in train", "left")]


def setup_font():
    """TT Commons Pro if installed, otherwise Segoe UI from Windows, otherwise matplotlib's DejaVu Sans."""
    home = Path.home()
    folders = [home / ".fonts", home / ".local/share/fonts", Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
               Path("/mnt/c/Windows/Fonts"), *Path("/mnt/c/Users").glob("*/AppData/Local/Microsoft/Windows/Fonts")]
    for folder in folders:
        if not folder.is_dir():
            continue
        for pattern in ["*TTCommons*", "*TT Commons*", "*TT_Commons*", "segoeui*.ttf", "seguisb*.ttf"]:
            for font in folder.rglob(pattern) if not str(folder).startswith("/mnt/c") else folder.glob(pattern):
                font_manager.fontManager.addfont(str(font))
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = next(name for name in ["TT Commons Pro", "Segoe UI", "DejaVu Sans"] if name in installed)
    plt.rcParams.update({"font.family": family, "svg.fonttype": "path", "mathtext.fontset": "custom",
                         "mathtext.rm": family, "mathtext.it": f"{family}:italic", "mathtext.bf": f"{family}:bold", "mathtext.cal": family})
    return family


def new_canvas():
    figure = plt.figure(figsize=(16, 9), facecolor=WHITE)
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set_xlim(0, 160)
    axes.set_ylim(0, 90)
    axes.set_aspect("equal")
    axes.axis("off")
    return figure, axes


def line_height(size):
    return size * 1.32 / 7.2  # 1 unit = 0.1 inch = 7.2 pt


def text_block(ax, x, y, lines, ha="center"):
    """lines: (text, size, weight, color) tuples, stacked and vertically centered on y."""
    total = sum(line_height(size) for _, size, _, _ in lines)
    top = y + total / 2
    for text, size, weight, color in lines:
        height = line_height(size)
        ax.text(x, top - height / 2, text, ha=ha, va="center", fontsize=size, fontweight=weight, color=color, zorder=6)
        top -= height


def box(ax, x, y, w, h, kind, lines=(), radius=0.8, lw=1.2, z=3, fc=None):
    style = STYLES[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", fc=fc or style["fc"],
                                ec=style["ec"], ls=style["ls"], lw=lw, zorder=z))
    if lines:
        text_block(ax, x + w / 2, y + h / 2, lines)


def arrow(ax, points, label=None, label_xy=None, size=9, color=GRAY_11, lw=1.3, below=None):
    """Arrow along a polyline, head at the last point. The label sits on a white patch at label_xy,
    the optional second label below the last point's line."""
    ax.add_patch(FancyArrowPatch(path=MplPath(points), arrowstyle="-|>", mutation_scale=10, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=4, joinstyle="miter"))
    if label:
        x, y = label_xy or ((points[0][0] + points[-1][0]) / 2, (points[0][1] + points[-1][1]) / 2 + 1.1)
        ax.text(x, y, label, ha="center", va="center", fontsize=size, fontweight="bold", color=INK, zorder=7,
                bbox=dict(boxstyle="round,pad=0.12", fc=WHITE, ec="none"))
    if below:
        x, y = label_xy[0], points[-1][1] - (label_xy[1] - points[-1][1])
        ax.text(x, y, below, ha="center", va="center", fontsize=8, color=GRAY_9, zorder=7)


def line(ax, points, color=GRAY_11, lw=1.3):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, zorder=4, solid_capstyle="butt", solid_joinstyle="miter")


def lane_x(index):
    return LANE_X0 + index * LANE_PITCH


def lane_center(index):
    return lane_x(index) + LANE_W / 2


def margin_label(ax, y, title, details):
    lines = [(title, 11.5, "bold", INK)] + [(detail, 8.5, "normal", GRAY_9) for detail in details]
    total = sum(line_height(size) for _, size, _, _ in lines)
    top = y + total / 2
    for text, size, weight, color in lines:
        height = line_height(size)
        ax.text(2, top - height / 2, text, ha="left", va="center", fontsize=size, fontweight=weight, color=color)
        top -= height


def header(ax, title, subtitle):
    ax.text(2, 87.2, title, ha="left", va="center", fontsize=20, fontweight="bold", color=INK)
    ax.text(2, 83.6, subtitle, ha="left", va="center", fontsize=11, color=GRAY_9)


def input_row(ax, y, h):
    for index, lane in enumerate(LANES):
        x = lane_x(index)
        box(ax, x, y, LANE_W, h, "input")
        box(ax, x + 0.9, y + 0.9, 4.6, h - 1.8, "input", fc=LANE_COLORS[lane], radius=0.6, lw=0)
        ax.text(x + 3.2, y + h / 2, lane, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK, zorder=6)
        wrapped = textwrap.wrap(SENTENCES[lane], 36)
        text_block(ax, x + 6.6, y + h / 2, [(text, 8.5, "normal", INK) for text in wrapped], ha="left")


def lane_arrows(ax, y_from, y_to):
    for index in range(len(LANES)):
        arrow(ax, [(lane_center(index), y_from), (lane_center(index), y_to)])


def concat_bar(ax, y, h, segments):
    """segments: (x_from, x_to, color, label) from left to right, one bar with dividers."""
    for x_from, x_to, color, label in segments:
        ax.add_patch(Rectangle((x_from, y), x_to - x_from, h, fc=color, ec=INK, lw=1.2, zorder=3))
        ax.text((x_from + x_to) / 2, y + h / 2, label, ha="center", va="center", fontsize=9, fontweight="bold",
                color=INK, zorder=6)


def classifier_row(ax, y, h, ops, shapes, bar_bottom, bar_x):
    """ops: (kind, lines, width) from left to right over the lanes, shapes: label of every arrow between them.
    A light funnel connects the whole concat bar to the first op."""
    span = (lane_x(3) + LANE_W) - LANE_X0
    gap = (span - sum(width for _, _, width in ops)) / (len(ops) - 1)
    x = LANE_X0
    first_w = ops[0][2]
    ax.add_patch(Polygon([(bar_x[0], bar_bottom), (bar_x[1], bar_bottom), (LANE_X0 + first_w, y + h), (LANE_X0, y + h)],
                         closed=True, fc=GRAY_1, ec="none", zorder=1))
    for index, (kind, lines, width) in enumerate(ops):
        box(ax, x, y, width, h, kind, lines)
        if index < len(ops) - 1:
            above, below = shapes[index] if isinstance(shapes[index], tuple) else (shapes[index], None)
            arrow(ax, [(x + width, y + h / 2), (x + width + gap, y + h / 2)], above,
                  label_xy=(x + width + gap / 2, y + h / 2 + 1.6), below=below)
        x += width + gap


def text_width(ax, text):
    """Rendered width of a text artist in data units (1 unit = 0.1 inch)."""
    return text.get_window_extent(ax.figure.canvas.get_renderer()).width / (ax.figure.dpi * 0.1)


def legend(ax, y, notes):
    items = [("trainable", GREEN, INK, "-"), ("not trained (rules, frozen GloVe, fixed idf)", GRAY_2, GRAY_7, "-"),
             ("no weights", WHITE, GRAY_7, (0, (3, 2))), ("observation", BLUE, INK, "-"),
             ("hypothesis 1", PURPLE, INK, "-"), ("hypothesis 2", YELLOW, INK, "-")]
    x = 2
    for label, fc, ec, ls in items:
        ax.add_patch(FancyBboxPatch((x, y - 0.9), 3, 1.8, boxstyle="round,pad=0,rounding_size=0.4", fc=fc, ec=ec,
                                    ls=ls, lw=1.1))
        text = ax.text(x + 3.8, y, label, ha="left", va="center", fontsize=9, color=INK)
        x += 3.8 + text_width(ax, text) + 3.5
    for index, note in enumerate(notes):
        ax.text(2, y - 3.0 - index * 1.9, note, ha="left", va="center", fontsize=8.5, color=GRAY_9)


def save(figure, name):
    for suffix in ["png", "svg"]:
        figure.savefig(OUT_DIR / f"{name}.{suffix}", dpi=200, facecolor=WHITE)
    plt.close(figure)
    print(f"wrote {OUT_DIR / name}.png and .svg")


def static_diagram():
    figure, ax = new_canvas()
    header(ax, "Static model · GloVe TF-IDF mean + 2-layer MLP",
           "glove_tfidf_mlp  ·  153,986 trainable parameters  ·  shapes per example, from raw text to the answer")

    # 1 input text
    margin_label(ax, 74.5, "Input", ["train example 0", "label: H2 is the better explanation"])
    input_row(ax, 70, 9)
    lane_arrows(ax, 70, 66.5)

    # 2 preprocessing
    margin_label(ax, 62, "Preprocess", ["nltk word_tokenize, drop punctuation,", "lowercase, keep stopwords"])
    for index, lane in enumerate(LANES):
        count = len(TOKENS[lane].split())
        lines = [(text, 8, "normal", INK) for text in textwrap.wrap(TOKENS[lane], 44)]
        box(ax, lane_x(index), 57.5, LANE_W, 9, "fixed", lines + [(f"{count} tokens", 9, "bold", INK)])
    lane_arrows(ax, 57.5, 54.5)

    # 3 GloVe lookup
    margin_label(ax, 51.75, "GloVe lookup", ["glove-wiki-gigaword-300, frozen", "400,000 words × 300 dims"])
    for index, lane in enumerate(LANES):
        count = len(TOKENS[lane].split())
        box(ax, lane_x(index), 49, LANE_W, 5.5, "fixed",
            [(f"{count} × 300", 10, "bold", INK), ("one word vector per token", 8, "normal", GRAY_9)])
    lane_arrows(ax, 49, 46)

    # 4 tf-idf weighted mean
    margin_label(ax, 43.25, "TF-IDF weighted mean", ["Σ idf(t) · v(t) / Σ idf(t)", "idf fitted on train, fixed"])
    for index in range(len(LANES)):
        box(ax, lane_x(index), 40.5, LANE_W, 5.5, "fixed",
            [("300", 10, "bold", INK), ("sentence vector", 8, "normal", GRAY_9)])
    lane_arrows(ax, 40.5, 37)

    # 5 concatenation
    margin_label(ax, 35.25, "Concatenate", ["fixed order O1, O2, H1, H2", "→ 1200 dims"])
    bar_x = (LANE_X0 - 1, lane_x(3) + LANE_W + 1)
    concat_bar(ax, 33.5, 3.5, [(lane_x(i) - 1, lane_x(i) + LANE_W + 1, LANE_COLORS[lane], f"{lane} · 300")
                               for i, lane in enumerate(LANES)])

    # 6 classifier
    margin_label(ax, 21, "2-layer MLP", ["trainable, 153,986 params", "Linear → ReLU → Linear"])
    ops = [
        ("no_weights", [("Dropout", 10.5, "bold", INK), ("p = 0.3", 8.5, "normal", GRAY_9)], 11),
        ("trainable", [("Linear", 10.5, "bold", INK), ("1200 → 128", 9.5, "bold", INK),
                       ("153,728 params", 8.5, "normal", INK)], 17),
        ("no_weights", [("ReLU", 10.5, "bold", INK)], 8),
        ("no_weights", [("Dropout", 10.5, "bold", INK), ("p = 0.3", 8.5, "normal", GRAY_9)], 11),
        ("trainable", [("Linear", 10.5, "bold", INK), ("128 → 2", 9.5, "bold", INK),
                       ("258 params", 8.5, "normal", INK)], 15),
        ("no_weights", [("Softmax", 10.5, "bold", INK), ("p(H1) + p(H2) = 1", 8.5, "normal", GRAY_9)], 15),
        ("input", [("p(H1), p(H2)", 10.5, "bold", INK), ("answer = argmax", 8.5, "normal", GRAY_9),
                   ("loss = CrossEntropy vs. label", 8.5, "normal", GRAY_9)], 22),
    ]
    classifier_row(ax, 16, 10, ops, ["1200", "128", "128", "128", ("2", "logits"), "2"], 33.5, bar_x)

    legend(ax, 9.5, [
        "TF-IDF example H1 \"he was never lost\": he 19.3 %, was 14.9 %, never 30.7 %, lost 35.1 % of the sentence vector "
        "(plain mean: 25 % each). Features are built once with numpy, no gradient reaches GloVe or the idf.",
        "Training: AdamW lr 1e-4, batch 64 (every shape gets a leading batch dimension B), early stopping on loss/valid "
        "with patience 5, best checkpoint by loss/valid.",
    ])
    save(figure, "static_glove_tfidf_mlp")


def story_block(ax, x0, width, name, sequence, colors, h_name, h_color):
    """Unrolled bidirectional LSTM over one candidate story of 3 sentence vectors."""
    y_fwd, y_in, y_bwd, cell_h, cell_w = 51.5, 44.5, 37.5, 4.6, 9
    box(ax, x0, 31, width, 28.5, "container", radius=1.2)
    ax.text(x0 + 1.5, 57.3, f"Story {name}  ·  {' → '.join(sequence)}", ha="left", va="center", fontsize=10.5,
            fontweight="bold", color=INK)
    ax.text(x0 + width - 1.5, 57.3, "same LSTM weights", ha="right", va="center", fontsize=8.5, color=GRAY_9)
    for label, y in [("forward", y_fwd), ("input", y_in), ("backward", y_bwd)]:
        ax.text(x0 + 1.5, y, label, ha="left", va="center", fontsize=8, color=GRAY_9)

    centers = [x0 + 13, x0 + 27, x0 + 41]
    for step, (center, sentence, color) in enumerate(zip(centers, sequence, colors)):
        left = center - cell_w / 2
        box(ax, left, y_in - cell_h / 2, cell_w, cell_h, "input", fc=color, radius=0.5, lw=1.1,
            lines=[(f"{sentence} · 300", 8.5, "bold", INK)])
        box(ax, left, y_fwd - cell_h / 2, cell_w, cell_h, "trainable", radius=0.5, lines=[("LSTM →", 8.5, "bold", INK)])
        box(ax, left, y_bwd - cell_h / 2, cell_w, cell_h, "trainable", radius=0.5, lines=[("LSTM ←", 8.5, "bold", INK)])
        arrow(ax, [(center, y_in + cell_h / 2), (center, y_fwd - cell_h / 2)])
        arrow(ax, [(center, y_in - cell_h / 2), (center, y_bwd + cell_h / 2)])
        if step < 2:  # state handed to the next time step: forward to the right, backward to the left
            arrow(ax, [(center + cell_w / 2, y_fwd), (centers[step + 1] - cell_w / 2, y_fwd)],
                  "256" if step == 0 else None, label_xy=(center + 7, y_fwd + 1.4), size=7.5)
            arrow(ax, [(centers[step + 1] - cell_w / 2, y_bwd), (center + cell_w / 2, y_bwd)],
                  "256" if step == 0 else None, label_xy=(center + 7, y_bwd + 1.4), size=7.5)

    # h = [h_forward after O2 ; h_backward after O1], 2 x 256
    bar_left, bar_w, bar_bottom, bar_top = x0 + 50.5, 3, 34.5, 54
    middle = (bar_bottom + bar_top) / 2
    ax.add_patch(Rectangle((bar_left, middle), bar_w, bar_top - middle, fc=h_color, ec=INK, lw=1.1, zorder=3))
    ax.add_patch(Rectangle((bar_left, bar_bottom), bar_w, middle - bar_bottom, fc=h_color, ec=INK, lw=1.1, zorder=3))
    arrow(ax, [(centers[2] + cell_w / 2, y_fwd), (bar_left, y_fwd)], "h→", label_xy=(x0 + 48, y_fwd + 1.4), size=8)
    arrow(ax, [(centers[0], y_bwd - cell_h / 2), (centers[0], 33), (x0 + 48.5, 33), (x0 + 48.5, 39), (bar_left, 39)])
    ax.text(x0 + 46.9, 40.6, "h←", ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
    text_block(ax, bar_left + bar_w + 3.1, middle, [(h_name, 9.5, "bold", INK), ("512", 8.5, "bold", INK),
                                                   ("256 + 256", 7.5, "normal", GRAY_9)])
    return bar_left + bar_w / 2, bar_bottom


def rnn_diagram():
    figure, ax = new_canvas()
    header(ax, "RNN model · candidate stories + shared bidirectional LSTM + MLP",
           "glove_lstm_bidirectional_improved  ·  1,405,698 trainable parameters  ·  shapes per example, from raw text "
           "to the answer")

    # 1 input text
    margin_label(ax, 76, "Input", ["train example 0", "label: H2 is the better explanation"])
    input_row(ax, 72, 8)
    lane_arrows(ax, 72, 68.5)

    # 2 sentence vectors, same as the static model
    margin_label(ax, 66, "Sentence vectors", ["same as the static model: preprocess,", "GloVe 300 (frozen), TF-IDF mean"])
    for index in range(len(LANES)):
        box(ax, lane_x(index), 63.5, LANE_W, 5, "fixed", [("300", 10, "bold", INK), ("sentence vector", 8, "normal", GRAY_9)])
        line(ax, [(lane_center(index), 63.5), (lane_center(index), 61.6)])
    line(ax, [(lane_center(0), 61.6), (lane_center(3), 61.6)])
    story_a_x, story_b_x, story_w = LANE_X0, lane_x(2), 2 * LANE_W + (LANE_PITCH - LANE_W)
    for x in [story_a_x, story_b_x]:
        arrow(ax, [(x + story_w / 2, 61.6), (x + story_w / 2, 59.5)])

    # 3 candidate framing + shared bidirectional LSTM
    margin_label(ax, 45, "Shared BiLSTM", ["trainable, 1,142,784 params", "1 layer, 300 → 256 per direction",
                                           "one LSTM reads both stories", "", "h→ forward state after O2",
                                           "h← backward state after O1", "cell state c stays internal"])
    out_a = story_block(ax, story_a_x, story_w, "A", ["O1", "H1", "O2"], [BLUE, PURPLE, BLUE], "h_A", PURPLE)
    out_b = story_block(ax, story_b_x, story_w, "B", ["O1", "H2", "O2"], [BLUE, YELLOW, BLUE], "h_B", YELLOW)

    # 4 concatenation of both stories
    margin_label(ax, 27.25, "Concatenate", ["[h_A ; h_B] → 1024 dims"])
    bar_x = (LANE_X0 - 1, lane_x(3) + LANE_W + 1)
    split = (story_a_x + story_w + story_b_x) / 2
    concat_bar(ax, 26, 2.5, [(bar_x[0], split, PURPLE, "h_A · 512  =  [h→ ; h←] of story A"),
                             (split, bar_x[1], YELLOW, "h_B · 512  =  [h→ ; h←] of story B")])
    for x, bottom in [out_a, out_b]:
        arrow(ax, [(x, bottom), (x, 28.5)])

    # 5 classifier head
    margin_label(ax, 17.5, "MLP head", ["trainable, 262,914 params", "Linear → ReLU → Linear"])
    ops = [
        ("no_weights", [("Dropout", 10.5, "bold", INK), ("p = 0.1", 8.5, "normal", GRAY_9)], 11),
        ("trainable", [("Linear", 10.5, "bold", INK), ("1024 → 256", 9.5, "bold", INK),
                       ("262,400 params", 8.5, "normal", INK)], 17),
        ("no_weights", [("ReLU", 10.5, "bold", INK)], 8),
        ("no_weights", [("Dropout", 10.5, "bold", INK), ("p = 0.1", 8.5, "normal", GRAY_9)], 11),
        ("trainable", [("Linear", 10.5, "bold", INK), ("256 → 2", 9.5, "bold", INK),
                       ("514 params", 8.5, "normal", INK)], 15),
        ("no_weights", [("Softmax", 10.5, "bold", INK), ("p(H1) + p(H2) = 1", 8.5, "normal", GRAY_9)], 15),
        ("input", [("p(H1), p(H2)", 10.5, "bold", INK), ("answer = argmax", 8.5, "normal", GRAY_9),
                   ("loss = CrossEntropy vs. label", 8.5, "normal", GRAY_9)], 22),
    ]
    classifier_row(ax, 12.5, 10, ops, ["1024", "256", "256", "256", ("2", "logits"), "2"], 26, bar_x)

    legend(ax, 7.2, [
        "Training: AdamW lr 1e-3, batch 64 (every shape gets a leading batch dimension B), early stopping on loss/valid "
        "with patience 5, best checkpoint by loss/valid.",
    ])
    save(figure, "rnn_glove_lstm_bidirectional_improved")


def bar(ax, x0, y, length, height, color):
    """Horizontal bar anchored at x0: square at the baseline, rounded at the data end."""
    ax.add_patch(FancyBboxPatch((x0, y - height / 2), length, height, boxstyle="round,pad=0,rounding_size=0.5",
                                fc=color, ec="none", zorder=3))
    ax.add_patch(Rectangle((x0, y - height / 2), min(0.6, length), height, fc=color, ec="none", zorder=3))


def tfidf_diagram():
    figure, ax = new_canvas()
    header(ax, "TF-IDF weighted mean · how tokens are weighted before averaging",
           "Example: H1 \"He was never lost.\" of train example 0  ·  same weighting in both final models  ·  "
           "fixed numbers, not learned")

    rows = [64, 55, 46, 37]
    idf_x0, idf_unit = 43, 3.4  # units per idf point
    share_x0, share_unit = 95, 0.8  # units per percent
    sigma_x, sigma_y, sigma_r = 139.5, sum(rows) / len(rows), 3
    total_idf = sum(idf for _, idf, _ in TFIDF_EXAMPLE)

    columns = [
        (2, "Token", ["after preprocessing"]),
        (19, "Word vector v(t)", ["GloVe, frozen, 300 dims"]),
        (idf_x0, "idf(t): how rare is the word?", [f"ln((1 + n) / (1 + df)) + 1,  n = {TRAIN_SENTENCES:,} sentences"]),
        (share_x0, "Weight w(t) = idf(t) / Σ idf", ["dashed line: plain mean, 25 % each"]),
        (sigma_x - sigma_r, "Weighted mean", ["s = Σ w(t) · v(t)"]),
    ]
    for x, title, details in columns:
        text_block(ax, x, 74.5, [(title, 11.5, "bold", INK)] + [(d, 8.5, "normal", GRAY_9) for d in details], ha="left")

    # recessive baselines and the plain-mean reference
    for x in [idf_x0, share_x0]:
        line(ax, [(x, 33.5), (x, 67.5)], color=GRAY_4, lw=1)
    reference_x = share_x0 + 25 * share_unit
    ax.plot([reference_x, reference_x], [33.5, 67.5], color=GRAY_9, lw=1.2, ls=(0, (4, 3)), zorder=5)
    ax.text(reference_x, 68.8, "25 %", ha="center", va="center", fontsize=8.5, color=GRAY_9)

    for y, (token, idf, df) in zip(rows, TFIDF_EXAMPLE):
        share = idf / total_idf
        box(ax, 2, y - 3, 13, 6, "input", [(token, 12, "bold", INK)])
        arrow(ax, [(15, y), (19, y)])
        box(ax, 19, y - 3, 18, 6, "fixed", [(f"v({token})", 10.5, "bold", INK), ("300", 8.5, "normal", GRAY_9)])

        bar(ax, idf_x0, y, idf * idf_unit, 3.6, PURPLE)
        value = ax.text(idf_x0 + idf * idf_unit + 1, y, f"{idf:.2f}", ha="left", va="center", fontsize=10.5,
                        fontweight="bold", color=INK)
        ax.text(idf_x0 + idf * idf_unit + 1 + text_width(ax, value) + 1.2, y,
                f"in {df:,} sentences ({df / TRAIN_SENTENCES:.1%})".replace("%", " %"), ha="left", va="center",
                fontsize=8.5, color=GRAY_9)

        bar(ax, share_x0, y, share * 100 * share_unit, 3.6, PURPLE)
        ax.text(share_x0 + share * 100 * share_unit + 0.6, y, f"{share:.1%}".replace("%", " %"), ha="left",
                va="center", fontsize=10.5, fontweight="bold", color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc=WHITE, ec="none"))  # in front of the dashed reference

        # every weighted word vector flows into the sum
        start = (131.5, y)
        dx, dy = sigma_x - start[0], sigma_y - y
        distance = (dx ** 2 + dy ** 2) ** 0.5
        arrow(ax, [start, (sigma_x - dx / distance * sigma_r, sigma_y - dy / distance * sigma_r)])

    # sums below the columns
    for x0, x1 in [(idf_x0, idf_x0 + 40), (share_x0, share_x0 + 32)]:
        line(ax, [(x0, 32), (x1, 32)], color=GRAY_7, lw=1)
    ax.text(2, 29.8, f"{len(TFIDF_EXAMPLE)} tokens", ha="left", va="center", fontsize=9, color=GRAY_9)
    ax.text(idf_x0, 29.8, f"Σ idf = {total_idf:.2f}", ha="left", va="center", fontsize=10.5, fontweight="bold",
            color=INK)
    ax.text(share_x0, 29.8, "Σ w = 100 %", ha="left", va="center", fontsize=10.5, fontweight="bold", color=INK)

    ax.add_patch(Circle((sigma_x, sigma_y), sigma_r, fc=WHITE, ec=INK, lw=1.4, zorder=5))
    ax.text(sigma_x, sigma_y + 0.15, "Σ", ha="center", va="center", fontsize=17, fontweight="bold", color=INK, zorder=6)
    arrow(ax, [(sigma_x + sigma_r, sigma_y), (146, sigma_y)])
    box(ax, 146, sigma_y - 6.5, 12, 13, "fixed", [("s", 13, "bold", INK), ("300", 10.5, "bold", INK),
                                                  ("sentence", 8.5, "normal", GRAY_9), ("vector", 8.5, "normal", GRAY_9)])

    # where the example words sit on the idf scale of the whole training vocabulary
    margin_label(ax, 16.5, "idf scale", ["all 16,189 training words", "left: common, low weight",
                                         "right: rare, high weight"])
    scale_x0, scale_x1, scale_min, scale_max, scale_y = 43, 156, 2, 12.5, 16.5
    position = lambda idf: scale_x0 + (idf - scale_min) / (scale_max - scale_min) * (scale_x1 - scale_x0)
    line(ax, [(scale_x0, scale_y), (scale_x1, scale_y)], color=GRAY_4, lw=2)
    for token, idf, _ in TFIDF_EXAMPLE:
        ax.add_patch(Circle((position(idf), scale_y), 0.75, fc=PURPLE, ec=WHITE, lw=1.5, zorder=5))
        text_block(ax, position(idf), scale_y + 3.2, [(token, 9.5, "bold", INK), (f"{idf:.2f}", 8.5, "normal", GRAY_9)])
    for token, idf, note, align in IDF_CONTEXT:
        ax.add_patch(Circle((position(idf), scale_y), 0.75, fc=GRAY_7, ec=WHITE, lw=1.5, zorder=5))
        x = position(idf) + {"center": 0, "right": 0.8, "left": -0.8}[align]
        text_block(ax, x, scale_y - 3.2, [(f"{token} {idf:.2f}", 9, "bold", GRAY_9), (note, 8, "normal", GRAY_9)],
                   ha=align)

    for index, note in enumerate([
        "The idf is computed once on the training sentences, before training. Backprop never reaches it, so the model "
        "cannot learn which words matter. A token that appears twice in a sentence counts twice.",
        "Plain mean for comparison: every token gets 1 / 4 = 25 %. With tf-idf the rare content words never and lost "
        "carry two thirds of the sentence vector.",
    ]):
        ax.text(2, 5.2 - index * 1.9, note, ha="left", va="center", fontsize=8.5, color=GRAY_9)
    save(figure, "tfidf_weighting")


def output_diagram():
    """Only what the code configures: the last Linear layer, its logits and the three library calls that use them."""
    figure, ax = new_canvas()
    header(ax, "Output format · what the code configures",
           "torch + Lightning, the same in both final models  ·  B = batch size (64)")

    # last layer -> logits
    trunk_y, trunk_x = 47, 50
    box(ax, 2, trunk_y - 7, 22, 14, "trainable", [("nn.Linear(hidden, 2)", 10.5, "bold", INK),
                                                   ("output_dim = 2", 8.5, "normal", INK),
                                                   ("static 128 → 2, RNN 256 → 2", 8.5, "normal", INK)])
    arrow(ax, [(24, trunk_y), (29, trunk_y)])
    box(ax, 29, trunk_y - 7, 16, 14, "input", [("logits", 11, "bold", INK), ("(B, 2)", 10.5, "bold", INK),
                                               ("column 0 = H1", 8.5, "normal", GRAY_9),
                                               ("column 1 = H2", 8.5, "normal", GRAY_9)])
    line(ax, [(45, trunk_y), (trunk_x, trunk_y)])

    # target, used by loss and accuracy
    box(ax, 2, 18, 43, 11, "input", [("y = label − 1", 11, "bold", INK), ("dataset label 1 / 2  →  target 0 / 1", 8.5, "normal", GRAY_9),
                                     ("tokenize_split, used by the loss and the accuracy", 8.5, "normal", GRAY_9)])

    branches = [
        (67, "Training  ·  every step", "nn.CrossEntropyLoss()(logits, y)", "softmax is inside the loss, the model returns raw logits",
         "loss/train", "minimized by AdamW"),
        (47, "Validation + test  ·  every epoch", "(logits.argmax(dim=1) == y).float().mean()", "share of correct answers, plus the same loss",
         "accuracy/valid · accuracy/test", "EarlyStopping + ModelCheckpoint watch loss/valid"),
        (27, "Prediction  ·  predict_step, RNN notebook", "torch.softmax(logits, dim=1)", "(B, 2), every row sums to 1",
         "p(H1), p(H2)", "answer = argmax + 1  →  1 or 2, like the label"),
    ]
    for y, title, code, code_note, result, result_note in branches:
        ax.text(56, y + 6.6, title, ha="left", va="center", fontsize=11.5, fontweight="bold", color=INK)
        arrow(ax, [(trunk_x, trunk_y), (trunk_x, y), (56, y)])
        box(ax, 56, y - 4.5, 52, 9, "no_weights", [(code, 10.5, "bold", INK), (code_note, 8.5, "normal", GRAY_9)])
        arrow(ax, [(108, y), (114, y)])
        box(ax, 114, y - 4.5, 42, 9, "input", [(result, 11, "bold", INK), (result_note, 8.5, "normal", GRAY_9)])

    ax.text(2, 5, "Model classes inline in the final notebooks (shared_step, predict_step)  ·  targets: pipeline/preprocessing "
            "(tokenize_split)  ·  callbacks: pipeline/training.py (fit)  ·  answer: pipeline/evaluation.py",
            ha="left", va="center", fontsize=8.5, color=GRAY_9)
    save(figure, "output_format")


if __name__ == "__main__":
    print(f"font: {setup_font()}")
    static_diagram()
    rnn_diagram()
    tfidf_diagram()
    output_diagram()
