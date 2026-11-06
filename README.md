# Attention, in space

![Hovering the letter a in "temperate": layer 2 head 4 lights the space 7 letters back (82%), and a card lists its targets and the likely next letters](screenshots/shot-6-hover-v3.png)

A tiny character-level GPT, trained from scratch on Shakespeare, running entirely in the browser. Type, and watch its attention in 3D.

The letters you type sit along the bottom. Four layers stack upward. Inside each layer, a glowing arc runs from an earlier letter (one layer down) to a later letter (one layer up): the path information takes through the network. An arc's thickness and brightness show how much attention that letter pays. The top row shows the model's best guess at the next character after each position, brighter when it is more confident. Hover over a column, or tap it on a phone, to see that position's top-5 next characters.

It follows [micrograd 3D](https://github.com/joezgao/micrograd-3d): micrograd shows one tiny network learning; this shows a whole transformer reading.

## Open it

Open `attention.html` in a recent browser. It is a single 1.17 MB file with the model weights inlined; it needs an internet connection to load three.js and two fonts. If your browser blocks local files:

```bash
python3 -m http.server 8000   # then open http://localhost:8000/attention.html
```

- **Layer** (All, 1–4) and **Head** (1–4, Avg) pick which attention to draw.
- Hover over a letter, or tap it on a phone, to see where it looks: its targets light up in amber in proportion to their share, and a card lists its top targets (over the full 64-character context) next to its top next characters.
- **Tour** is five 8-second steps through the heads below, each following one example letter with its share computed live.
- **How it was made** (under the stats, top right) opens a short note on the model, the data and the training.
- Drag to orbit, scroll or pinch to zoom, double-click to reset.
- **Write** lets the model write (temperature 0.8). On arrival it writes for about 9 seconds, then waits for you.
- Narrow screens (under 720 px) show 16 letters instead of 32. `prefers-reduced-motion` turns off camera drift, scroll animation, easing and autoplay.

Measured on 4,096 held-out windows with the shipped weights, four heads stand out (the Tour shows exactly these):

| Head | Attends to | Share |
|---|---|---|
| Layer 1, head 1 | the two letters just before | 61% |
| Layer 2, head 4 | the last space: where the current word began | 81% |
| Layer 2, head 3 | the last line break | 58% |
| Layer 3, head 2 | vowels | 67% |

No head was found for an induction pattern, the same letter earlier, or the start of the line (none puts more than 6% of its attention there), so the page doesn't claim one.

## The model

| | Value |
|---|---|
| Architecture | nanoGPT design: learned token and position embeddings, 4 pre-LayerNorm blocks (causal self-attention plus a 4× MLP with tanh-GELU), a final LayerNorm, and an `lm_head` tied to the token embedding |
| Size | 4 layers, 4 heads, d_model 128, context 64 characters, vocab 65, **809,856 parameters** |
| Data | tiny-shakespeare (1,115,394 characters, from `karpathy/char-rnn`), split 90/10 into train and validation |
| Validation loss | **1.527** at the best checkpoint (step 6000, fp32). The shipped int8 weights score **1.531** on a larger fixed sample of validation windows; the page shows that number. |
| Shipped weights | int8 with one fp32 scale per row (845 kB), base64-inlined. Quantising changed the loss by +0.0003. |

**Training.** The model was written in pure NumPy with a hand-derived backward pass, following nanoGPT's `model.py` op for op (backward written layer by layer in the style of llm.c), and every parameter group was checked against finite differences. The recipe is nanoGPT's `train_shakespeare_char`, scaled down: AdamW (β 0.9, 0.99; weight decay 0.1 on matrices), lr 2e-3 with a 200-step warmup and cosine decay to 2e-4, gradient clipping at 1.0, batch 32 × 64 tokens, no dropout, 7,000 steps. It took 32 minutes on 2 CPU cores; validation loss was best at step 6000 and had started to overfit slightly by 7000 (1.530).

A sample at temperature 0.8, after the prompt *"ROMEO: But soft, what light through yonder window breaks? JULIET:"*:

> O Capulet, Aufidius,<br>
> And it is here loved so the state bird,<br>
> And set for yourself must this arty would show,<br>
> That which between her so his fingers<br>
> Upon him. But now heavens by this day,

## How the page works

- One HTML file. three.js 0.147.0 loads from jsDelivr; Lora and Inter load from Google Fonts. Everything else, weights included, is inline.
- The forward pass is plain JavaScript in an inline Web Worker, so the 3D view never waits for the model. Each new character re-runs the last 64 characters and returns every layer's attention matrices and the next-character probabilities.
- The arcs are instanced ribbons with widths computed in screen space in the vertex shader. Attention weights ease on the CPU; arcs too faint to see are dropped (usually about 400 of 2,112 remain), and the page only draws a frame when the picture changes.

## Limits

- 60 fps is not verified on a real GPU. It was only measured with software WebGL on 2 busy CPU cores: 10–14.5 fps at 1440 × 900 while writing (v1 managed 0.3–0.7). On a GPU the arcs are about 400 instanced ribbons (roughly 6k vertices) a frame, so 60 fps should be easy, but it needs checking on real hardware.
- It is a 0.8M-parameter model with 64 characters of context: mostly right spelling and the right form (speaker names, line breaks), but the meaning is nonsense.
- Only attention within the visible window is drawn; attention to characters that have scrolled out of view is not shown.
- Characters outside the training vocabulary (most digits, for example) are dropped from what you type.

## In this repository

- `attention.html`: the piece
- `screenshots/`: typing, generating, an orbited view, the phone layout, the Tour, hovering a letter (v3), a Tour step (v3) and the v3 phone layout

The NumPy training, gradient-check and export scripts are not in this repository yet.

## Credits and license

Model design after [nanoGPT](https://github.com/karpathy/nanoGPT) (MIT); training text: tiny-shakespeare from [char-rnn](https://github.com/karpathy/char-rnn) (Shakespeare, public domain). three.js is MIT; the fonts are under the SIL Open Font License.

Code: MIT. See [LICENSE](LICENSE).

Built with Claude for Joe Gao.
