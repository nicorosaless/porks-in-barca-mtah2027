# Porks in Barça: ball possession from automatic tracking

More Than a Hack 2027, Initial Challenge. Submission: `Porks in Barça.csv`
(`modelos/hibrido/it9b.json`, sha256 `97b40a031db486922d84bd973e78884d01a429d0053ae8c0f2c0183671d61156`).

## 1. Approach

The ATD is noisy. The ball is missing on 41 % of graded frames, often for many seconds, and the 46
track slots carry no identity. A frame-by-frame rule like "the team closest to the ball" breaks
whenever the ball disappears. Our pipeline therefore has two stages:

1. **Emission network**, which gives P(H, A, D) at 5 fps. Each step is a set of tokens: one per
   visible player (position, velocity, offset from the ball, team) and one for the ball. The ball
   token holds the observed or interpolated position, time since and until it was last seen, and
   whether it is out of the pitch. A small Transformer without positional encoding makes the
   output invariant to slot order, since slots in the ATD mean nothing. A bidirectional GRU then
   runs over time, so the network can use future frames (allowed, issue #15). About 0.6 M
   parameters (`mtah/hibrido/modelo.py`, `mtah/hibrido/rasgos.py`).
2. **Viterbi decoding** over the states H/A/D, with explicit costs for switching team, for going
   from live to dead and from dead to live, a bias on D, and a temperature on the emissions
   (`mtah/hibrido/decodificar.py`). This turns noisy per-step probabilities into possession spells,
   which is the unit `spellF1` scores. One spurious flicker is a false positive, and one missed
   change merges two possessions.

The final model averages the probabilities of two networks, 0.5 each, before Viterbi:

| component | training data | convention flag |
|---|---|---|
| `it4c.pt` | IDSSE only (7 matches) | none (DFL) |
| `it7.pt` | IDSSE + Metrica sample + SkillCorner (30 matches) | "Metrica style" at inference |

We chose this over a pure rule system or a single network because we measured it on external data
(section 5). Viterbi alone over a hand-made heuristic scored 0.34 on degraded IDSSE, and the network
plus Viterbi scored 0.72. The ensemble beat both components on challenge-style data and stayed
level with the best one on IDSSE.

**Bridging the domain gap without touching the provided game.** All external datasets have clean
ball tracking. We degrade them until they look like the ATD (`mtah/degradar.py`):
- ball missing on 40.7 % of graded frames, in long blocks;
- players fragmented into 46 anonymous slots, with about 20.5 visible;
- position noise;
- the DFL "frozen dead ball" artefact removed.

Without that last step, a network learns that a perfectly still ball means dead, which inflates
validation and does not transfer. Training mixes several degradation scenarios, including a
pessimistic one where the ball is never missing because play is dead.

## 2. Assets used and dropped

| asset | used? | how |
|---|---|---|
| Automatic tracking data (ATD) | **yes** | the only input to the network at inference |
| Smart Tagging | **only as a check** | tested as Viterbi potentials (team of the phase, restarts); rejected on external validation (section 5) |
| Homography / tactical video / broadcast video | no | the pipeline only uses tracking coordinates |
| Public 5-minute ground truth | **only as an alignment check** | looked at once per chosen model, never to choose, tune or train |
| FAQ.md fact: the second half is graded from frame 110,808 but kicks off at 113,742 | **yes** | hard rule: those frames are D |

Dropped after measuring:
- an IDSSE + Metrica joint model with a convention flag (it5);
- SkillCorner-heavy models alone (it6, it7), which were better on style data but lost too much on
  IDSSE;
- a 10-network seed ensemble (+0.006, within noise);
- minimum-duration constraints in Viterbi;
- Smart Tagging potentials;
- an IDSSE model trained on the Metrica convention;
- a "v4" degradation with extra ATD defects (duplicated players, fake ball, wrong team on
  re-acquisition, gaps on fast balls).
- test-time augmentation over the 8 pitch reflections and team swaps (−0.003).

## 3. Training data

**None of it comes from the provided game.** The BAR-ATH data, its Smart Tagging and its public
ground truth were never used to train or tune anything.

| dataset | matches | source and licence | who labelled it | possession labels we derive |
|---|---|---|---|---|
| IDSSE (DFL / Sportec Solutions) | 7 (German Bundesliga) | Bassek, Rein, Weber & Memmert (2025), *Scientific Data* 12, 195; CC BY 4.0 | DFL's official match data provider: `BallStatus` and `BallPossession` per frame | `BallStatus = 0` → D, otherwise the possessing team. Frames without `BallPossession` are not scored |
| Metrica Sports sample data | 3 | github.com/metrica-sports/sample-data (public, attribution requested) | Metrica's published event data | H/A/D derived from events: dead between stoppage events (out, foul, set piece) and the restart; team of the last on-ball event |
| SkillCorner open data | 20 (A-League 2024/25, broadcast tracking 10 fps, interpolated to 25) | github.com/SkillCorner/opendata, MIT | SkillCorner's `phases_of_play` and `dynamic_events` | D outside phases of play; team carried forward from `player_possession` events |

About 30 matches in total, roughly 45 hours of football. Every label is derived automatically from
the published data, and none is hand-made by us.

## 4. Possession rules

The target is the challenge's convention (CHALLENGE.md): one team has **control** of the ball.
Contested or loose phases keep the previous possessor, and D runs from the stoppage to the restart.

- **Loose balls and challenges.** Possession does not change on a touch, only on control. For IDSSE,
  whose labels change on every touch, we relabel in the "Metrica convention": a possession shorter
  than 2 s that comes straight from a live possession of the other team, with no D between them,
  is absorbed by its neighbour. We fixed that threshold on external data before looking at
  BAR-ATH: it matches the rate of possessions per live minute in Metrica's sample games. Viterbi's
  switching cost does the rest.
- **Restarts.** D starts when the ball goes dead and ends at the restart. Throw-ins, goal kicks,
  corners and free kicks are learnt from the external labels, not hand-coded.
- **Second-half kick-off.** Forced D from frame 110,808 to 113,742, as FAQ.md states.
- **Non-graded frames** (before kick-off, half-time, after the whistle) are written as D. The
  scorer ignores them.

## 5. Validation and model selection

We never had a scored signal on the full game, so every decision was taken on external data, with a
rule written down **before** each experiment ran (`docs/HIBRIDO.md` keeps the full log):
- **ZX-dev:** nested cross-validation on 16 challenge-style matches (Metrica and SkillCorner).
- **SI-M:** IDSSE relabelled in the Metrica convention, averaged over four degradation scenarios.
- **Safe box:** 9 matches held out and not looked at until the final freeze.
- **Acceptance:** an experiment had to gain at least +0.01–0.02 and win at least 13 of the 16
  matches.

| model | ZX-dev | SI-M | decision |
|---|---:|---:|---|
| it8 as first built (Viterbi costs fitted on IDSSE/DFL) | 0.646 | 0.680 | — |
| **it9b** (Viterbi costs fitted on challenge-style data) | **0.688** | **0.712** | accepted, 14/16 matches |
| + 10-network seed ensemble | 0.694 | 0.729 | rejected (9/16) |
| + minimum spell duration | 0.689 | 0.712 | rejected (6/16) |
| + Smart Tagging potentials (pessimistic tag noise) | 0.690 | — | rejected (8/16) |
| v4 degradation (IDSSE-only model, 3 seeds each arm) | +0.004 | +0.000 | rejected (within seed noise) |
| + test-time augmentation (8 reflections / team swaps) | 0.686 | 0.708 | rejected (7/16) |

Safe box (9 held-out matches), evaluated once at the freeze against it8 as first built: +0.051
on challenge-style data (0.677 vs 0.627, 6/7 matches) and +0.043 on IDSSE (0.722 vs 0.679, 2/2).

Public 5-minute window: spellF1 0.545 (alignment check only, about 17 possessions).

## 6. Limitations

- **Long ball gaps.** With no ball for several seconds, the network relies on player shape alone.
  Most remaining errors are merged or split possessions inside those gaps.
- **Short possessions** (< 3 s) are the weakest class. Viterbi favours longer spells, so the model
  predicts about 15 % fewer possessions than the ground truth on external data.
- **Domain gap.** No external dataset has the ATD's exact defects. Our degradation is an
  approximation, and the convention flag is a proxy for the Metrica labelling style.
- **With more time** we would add the ball's height from the video to separate aerial balls, train
  a restart detector on Smart Tagging-like labels from external data, and add labelled matches in
  the target convention.

## 7. Reproduction

```bash
git clone <this repository> && cd <repository>
# Put the challenge data bundle where DATA.md says (data/tactical/...), as for the starter kit.
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -r requirements-equipo.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
.venv/bin/python scripts/generar_entrega.py hibrido_it9b     # -> salidas/hibrido_it9b.csv
sha256sum salidas/hibrido_it9b.csv                          # 97b40a03…1156
```

This is inference only: CPU, a few minutes, no external data needed. The weights
(`modelos/hibrido/it4c.pt`, `it7.pt`, 2.4 MB each) are in the repository.

Retraining needs the external datasets (`scripts/descargar_skillcorner.py`, IDSSE from Hugging Face
`pysport/idsse-data`, Metrica sample data) and a GPU. The commands are in `docs/HIBRIDO.md`
(`scripts/hibrido/entrenar.py`, `afinar.py`, `ensamblar.py`).
