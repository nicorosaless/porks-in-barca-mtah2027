# Híbrido: primera exploración (7-nov-2026)

Tres iteraciones de hill climbing sobre la arquitectura híbrida del plan: emisiones por paso
(5 fps) más una máquina de estados H/A/D decodificada con Viterbi. Todo se ha elegido en IDSSE
degradado con validación cruzada anidada; el GT público no ha intervenido en ninguna decisión.

## Protocolo

- **Datos:** 7 partidos IDSSE, cada uno degradado con las semillas 0 y 1 (evaluación). Para
  entrenar se usan las semillas 1000–1005.
- **Red:** leave-one-match-out. Se entrena con 6 partidos y se predice el séptimo, con épocas
  fijas y sin parada temprana. Corre en una H100 de MareNostrum (`acc_debug`) con datos externos
  únicamente; cada run de 7 folds más el modelo final tarda unos 13 min.
- **Decodificador:** CV anidada (`scripts/hibrido/afinar.py`). Para cada partido, los costes se
  eligen por ascenso de coordenadas en los otros 6 y se puntúa con el scorer oficial.
- **Comprobación de alineación:** el GT público (5 min) se miró **una sola vez**, con el modelo ya
  elegido.

## Resultados (spellF1, CV anidada, 7 partidos × 2 semillas)

| iteración | qué cambia | deg. v1 | deg. v2 | pesimista |
|---|---|---|---|---|
| — | baseline smooth50 | 0.199 | 0.298 | 0.200 |
| 1 | heurística (cercanía al balón interpolado, fuera del campo, hueco) + Viterbi | 0.242 | 0.344 | 0.223 |
| 2 | red set-Transformer + BiGRU como emisiones | ~~0.648~~ (artefacto) | — | — |
| 3a | igual que 2, entrenada en degradación v2 | — | 0.696 | 0.541 |
| 3b | 3a + pérdida auxiliar de posición del balón | — | 0.694 | — |
| **3c** | 3a entrenada con **mezcla** de escenarios de balón muerto | — | **0.717** | **0.616** |

El macroF1 de 3c es 0.939 en v2 y 0.903 en el pesimista. La dispersión entre partidos es de
unos ±0.04.

**Comprobación de alineación de 3c en BAR-ATH (5 min públicos):** spellF1 0.497 y macroF1 0.803.
smooth50 saca 0.326 en esa misma ventana. La transferencia funciona, pero el resultado queda por
debajo de IDSSE. Entrega: `salidas/hibrido_it3c.csv`, sha256 `5446475a…e761`, 357 posesiones.

## Lo que se aprendió

1. **Hay un artefacto de DFL que infla la validación.** Con el balón muerto, IDSSE congela su
   posición: el desplazamiento es exactamente 0 en el 100 % de esos frames, frente al 0.1 % en el
   ATD. La iteración 2 lo aprendió y por eso su 0.648 no vale. `mtah/degradar.py` ya lo quita:
   cada tramo congelado se convierte en balón ausente o en movimiento lento sintético.
2. **Falta saber cuánto falta el balón cuando está muerto en el ATD, y es lo que más mueve el
   resultado.** Hay dos indicios a favor de que falta sobre todo cuando está muerto:
   - En el ATD se ve el balón en el 59.3 % de los frames. IDSSE tiene un 58.5 % de balón vivo.
   - El histograma de velocidad del ATD apenas tiene balón lento visible.

   Pero DATA.md dice que falta más en duelos. v2 calibra esto con agregados del input
   (`scripts/hibrido/calibrar_balon.py`) y el pesimista supone lo contrario. Entrenar con una
   mezcla de los dos escenarios (3c) es lo que mejor aguanta en ambos.
3. **El baseline también cambia mucho con esa suposición** (0.20 frente a 0.30). Los macroF1
   publicados del baseline en el partido completo (~0.67) se parecen más a v2 (0.72) que a v1
   (0.44). Esto se usó como comprobación, no como ajuste.
4. **Las convenciones de D son distintas.** En los frames evaluables, 3c predice ~45 % de balón
   muerto en BAR-ATH, que es la proporción de DFL. En los 5 min públicos el GT tiene un 20 %. O
   esa ventana es poco representativa o Metrica marca D más corto que DFL (issue #31). Es lo
   primero que hay que preguntar a los organizadores.

## Siguientes pasos para el equipo

- Preguntar por correo a los organizadores cuándo empieza y acaba D en el GT de Metrica.
- Modelar dónde faltan los huecos de balón (duelos, áreas), en lugar de colocarlos al azar.
- Usar el Smart Tagging como evidencia de reanudación en el decodificador (no existe en IDSSE;
  habría que simularlo).
- Comparar con más datos externos (PFF) para que la red no aprenda solo convenciones de DFL.

## Reproducir

```bash
export MTAH_EXTERNAL_DIR=/ruta/external
# MareNostrum (solo código + caches IDSSE): ver la cabecera de scripts/hibrido/entrenar_mn5.sbatch
RUNS="it3c:--mezcla --pesimista --final"
# local
python scripts/hibrido/afinar.py salidas/hibrido/runs/it3c [--sufijo _pes]
python scripts/generar_entrega.py hibrido_it3c
```

Los pesos (`modelos/hibrido/it3c.pt`, 2.4 MB) se entrenaron solo con IDSSE. Hay que citar IDSSE
(CC BY 4.0, ver docs/RESEARCH.md).

## it4: convención, huecos realistas y bordes

Se añadió Metrica Sports sample-data como segundo dominio externo. Fuente: `metrica-sports/sample-data`
en GitHub; el README no trae licencia formal y pide reconocer la fuente en usos públicos.

### Convención de posesión

La regla GT-M absorbe posesiones H/A cortas si vienen de una posesión viva del rival sin D entre
medias. T se fijó antes de mirar BAR-ATH: el valor de `{1,2,3}` que deja las posesiones por minuto
de balón vivo de IDSSE más cerca de Metrica sample. Ganó `T=3s`.

| fuente | variante | pos./parte | duración media | pos./min vivo | frames cambiados |
|---|---|---:|---:|---:|---:|
| IDSSE | original | 218.6 | 7.3 s | 8.38 | — |
| IDSSE | T=1s | 182.6 | 8.7 s | 7.00 | 0.45 % |
| IDSSE | T=2s | 144.1 | 11.0 s | 5.52 | 1.71 % |
| IDSSE | T=3s | 118.6 | 13.4 s | 4.55 | 3.11 % |
| Metrica sample | eventos | 173.3 | 12.7 s | 4.79 | — |

En Metrica sample, la fracción de balón NaN fue 0.391, 0.410 y 0.322 para SG1-SG3. El balón
congelado casi no aparece y no coincide con D, lo que refuerza quitar el artefacto de DFL.

### Resultados

`S` es la media de spellF1 en la CV anidada y `Z` el spellF1 zero-shot medio en Metrica sample.
Para P2/P3 se comparó `it4c` con la referencia DFL conjunta porque P1 fue rechazada.

| modelo | GT | costes | v2 | pes | v3 | v3pes | S | Z (Metrica) | macroF1 | pred/gt |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| it3c (HIBRIDO.md) | DFL | separados | 0.717 | 0.616 | — | — | 0.667 | — | 0.94/0.90 | — |
| it3c_conj | DFL | conjuntos | 0.716 | 0.616 | — | — | 0.666 | 0.276 | 0.921 | 0.78 |
| it3c_m | GT-M | conjuntos | 0.802 | 0.733 | — | — | 0.767 | 0.238 | 0.922 | 0.94 |
| it4a | GT-M | conjuntos | 0.797 | 0.732 | — | — | 0.764 | 0.221 | 0.923 | 0.91 |
| it4a_D4 | DFL | conjuntos | 0.563 | 0.505 | 0.552 | 0.507 | 0.532 | — | 0.904 | 0.54 |
| it4b | DFL | conjuntos | 0.718 | 0.615 | 0.695 | 0.611 | 0.660 | 0.282 | 0.919 | 0.76 |
| **it4c** | DFL | conjuntos | **0.725** | **0.616** | **0.701** | 0.609 | **0.663** | **0.313** | 0.915 | 0.81 |

Decisión:
- P1 se rechazó: `S(it4a)-S(it3c_m)=-0.003` y `Z(it4a)-Z(it3c_m)=-0.017`, así que no cumple la mejora de Z de +0.03 exigida cuando S empata.
- P2 no desplazó a la referencia DFL: `it4b` quedó dentro del margen de S y con Z insuficiente frente a P3.
- P3 aceptó `it4c`: frente a `it3c_conj`, la diferencia de S en v2/pes queda dentro de ±0.02 y Z mejora de 0.276 a 0.313 (+0.037).

### Comprobación final

Modelo elegido: `hibrido_it4c`. La comprobación de alineación contra el GT público se ejecutó una
sola vez, después de elegir el modelo: spellF1 0.5333, macroF1 0.8615, sha256
`ccf095483a424efe78acdfe2778632ddd81c24b1d37d4d27a587cd805064fc2e`.

Agregados de la predicción completa BAR-ATH: 362 posesiones, duración media 9.5 s, fracción D
0.568 y 29 % de posesiones de menos de 3 s. it3c tenía 357 posesiones y 27-30 % de menos de 3 s.

## it5: entrenamiento mixto IDSSE + Metrica

La hipótesis de it5 cambia respecto a it4a: no se reetiqueta DFL. Se entrena con IDSSE y Metrica
sample a la vez, y la red recibe una bandera constante de convención (`0 = DFL`, `1 = Metrica`)
para separar dominios. BAR-ATH se predice con la bandera Metrica.

### Fuente Metrica

Se corrigió la lectura de Metrica antes de entrenar. SG1 y SG2 tenían el tracking de jugadores
vacío por una cabecera CSV con tres filas; las cachés pesaban unos 600 KB y daban 0 jugadores
visibles por frame. Tras parsear los pares de columnas reales, las cachés quedan en ~22 MB y los
tres partidos tienen ~22 jugadores visibles por frame.

SG3 tenía pocas paradas porque el JSON anota varias reanudaciones como intervalos previos a
`SET PIECE`, y además usa `CHALLENGE ... FAULT`, tiros `OUT` y pérdidas `OFFSIDE/HAND BALL/GOAL KICK`
para eventos muertos. Esas reglas se aplican solo a SG3, para no cambiar la convención ya medida
en los CSV SG1/SG2.

| partido | jugadores visibles/frame | NaN jugadores | NaN balón | D antes | paradas antes | D después | paradas después |
|---|---:|---:|---:|---:|---:|---:|---:|
| SG1 | 0.0 -> 22.0 | n/a -> 0.214 | 0.391 | 0.338 | 78 | 0.338 | 78 |
| SG2 | 0.0 -> 22.0 | n/a -> 0.154 | 0.410 | 0.273 | 75 | 0.273 | 75 |
| SG3 | 22.0 | 0.371 | 0.322 | 0.087 | 30 | 0.276 | 98 |

La convención pública medida antes de decidir it5 fue:

| fuente | posesiones/min vivo | posesiones < 3 s |
|---|---:|---:|
| GT público del reto | 4.2 | 18 % |
| IDSSE (DFL) | 6.5-11 | 38-50 % |
| Metrica sample (etiquetas derivadas de eventos) | 4.3-5.5 | 23-31 % |

Al balón le falta en el 32-41 % de los frames en Metrica sample, frente al 41 % del ATD del reto.
Por eso Metrica se considera el dominio externo más cercano al reto; el GT público no se usa para
elegir ni ajustar.

### Estado de ejecución

El código implementa:
- `scripts/hibrido/entrenar.py --fuentes idsse,metrica --peso_metrica 2.0 --bandera_convencion`;
- degradación de Metrica sin quitar balón ni congelado (`balon_ausente=0`, `quitar_congelado=False`);
- OOF leave-one-match-out por fuente/partido;
- compatibilidad con modelos antiguos sin bandera;
- `scripts/hibrido/afinar.py --fuente_datos metrica` para el ZL de Metrica.

`acc_debug` estaba ocupada por un trabajo retenido de otro proyecto, así que it5 se lanzó en
`acc_training` (job 48808342, 20 min). Ojo: `sbatch --export` separa por comas, así que dentro
de `RUNS` las listas van con `+` (`--fuentes idsse+metrica`). Un primer intento con comas
entrenó otra cosa y se descartó.

ZL: spellF1 de la CV anidada en Metrica sample, degradación de Metrica con semillas 0 y 1. Los
costes de cada partido se ajustan en los otros 2. it3c e it4c se evalúan zero-shot con
`scripts/hibrido/predecir_externo.py`.

| modelo | ZL (SG1 / SG2 / SG3) | ZL media | macroF1 | decisión |
|---|---|---:|---:|---|
| it3c | 0.776 / 0.789 / 0.728 | 0.764 | 0.865 | referencia |
| it4c | 0.784 / 0.769 / 0.739 | 0.764 | 0.863 | empata con it3c |
| it5 | 0.825 / 0.798 / 0.716 | 0.780 | 0.869 | **rechazado**: +0.016 < +0.03 |

**La Z de it4 era inválida.** Se calculó con SG1/SG2 sin jugadores, por el bug del parser
(~0.3 en lugar de ~0.76). Con Metrica bien leída, it4c e it3c empatan en Metrica (0.764) y
en IDSSE (dentro de ±0.02). Por la regla de la sección 4 del plan, it4c no debió sustituir a
it3c. La diferencia entre los dos está dentro del ruido.

El control it5_dfl no se lanzó: el zero-shot de it3c/it4c ya cubre «entrenado solo con IDSSE».
SI de it5 no se calculó, porque it5 no supera el umbral de ZL.

No se ejecutó `scripts/generar_entrega.py`: al no haber modelo it5 entrenado y seleccionado, la
única comprobación pública válida sigue siendo la de it4c.

## it6: SkillCorner open data (20 partidos A-League 24/25)

Fuente: `github.com/SkillCorner/opendata` (MIT; citar a SkillCorner). Tracking de retransmisión a
10 fps interpolado a 25 (`mtah/fuentes/skillcorner.py`). El balón y los jugadores no detectados
quedan en NaN. Al balón le falta en el 35–48 % de los frames (ATD: 41 %); se ven unos 9 jugadores
por frame (ATD: ~20).

Etiquetas: D fuera de `phases_of_play`. El equipo sale de los eventos `player_possession` con
forward-fill, porque `possession.group` del tracking solo coincidía con esos eventos en un 85 %
(los eventos, 97 %). Agregados: 4,9 posesiones por minuto de balón vivo, 20 % de posesiones de
menos de 3 s (Metrica: 4,3–5,5 y 23–31 %), 46 % de D (Metrica: ~30 %).

Validación: `--kfold 6` estratificado por fuente sobre 30 partidos. ZX es la CV anidada de los 23
partidos de estilo reto (3 de Metrica y 20 de SkillCorner), escenario v2, semillas 0 y 1. SI es
IDSSE en v2/pes/v3/v3pes. Las evaluaciones se lanzaron con `--procesos 3`: con 8 procesos se
quedaron sin memoria.

| modelo | ZX | Metrica | SkillCorner | ganados a it4c | SI (IDSSE) |
|---|---:|---:|---:|---:|---:|
| it4c | 0.616 | 0.705 | 0.603 | — | 0.663 |
| it5 | 0.540 | 0.700 | 0.515 | 2/23 | — |
| it6 | **0.685** | 0.723 | 0.679 | **23/23** | **0.574** |

SI de it6 por escenario: v2 0.666, pes 0.508, v3 0.633, v3pes 0.490.

**Decisión según la regla fijada: it6 se rechaza.** Cumple ZX (+0.068 ≥ +0.02) y gana en 23/23
partidos, pero `SI(it6) = 0.574 < SI(it4c) − 0.03 = 0.633`. Sigue vigente it4c y no se hace
comprobación pública. Con el mismo dato, el criterio no se cambia a posteriori.

## it7: it6 sin el atajo «sin balón = muerto» (prerregistro, 2026-11-11)

Hipótesis: it6 pierde en IDSSE (también en v2: 0.666 frente a 0.725) porque en la retransmisión
de SkillCorner el balón falta sobre todo en las paradas (repeticiones, planos cortos), y la red
aprende que la falta de balón indica balón muerto. En el ATD el balón falta un 41 % de los
frames con el juego vivo o parado. Además, el estilo reto pesa ~87 % del entrenamiento de it6.

Cambios frente a it6 (todo lo demás igual: `--kfold 6`, v3, `--mezcla`, borde 3, bandera):
- `--huecos_estilo 0+0.5+0.6`: en Metrica y SkillCorner, las semillas de entrenamiento rotan
  entre ningún hueco extra y huecos aleatorios hasta el 50 % o el 60 % de frames sin balón.
  Los huecos solo empiezan en frames con balón visible, así que caen sobre todo en juego vivo.
- `--peso_estilo 1.0` (it6: 2.0).

Regla (la misma que it6, fijada antes de entrenar): se acepta it7 si
`ZX(it7) − ZX(it4c) ≥ +0.02` **y** it7 gana a it4c en ≥ 13 de 23 partidos **y**
`SI(it7) ≥ SI(it4c) − 0.03 = 0.633`. Si se acepta, se mira el GT público una vez.

### Resultado de it7

MareNostrum, job 48933319 (13 min, tras ~2,5 h en cola). Evaluación local con `--procesos 3`.

| modelo | ZX | Metrica | SkillCorner | ganados a it4c | SI (IDSSE) | SI v2 / pes / v3 / v3pes |
|---|---:|---:|---:|---:|---:|---|
| it4c | 0.616 | 0.705 | 0.603 | — | 0.663 | 0.725 / 0.616 / 0.701 / 0.609 |
| it6 | 0.685 | 0.723 | 0.679 | 23/23 | 0.574 | 0.666 / 0.508 / 0.633 / 0.490 |
| it7 | 0.678 | 0.698 | 0.675 | 21/23 | **0.589** | 0.671 / 0.526 / 0.637 / 0.521 |

**Decisión según la regla fijada: it7 se rechaza.** Cumple ZX (+0.062) y gana en 21/23, pero
`SI(it7) = 0.589 < 0.633`. Sigue vigente it4c.

Lectura: quitar el atajo y bajar el peso del estilo apenas recupera IDSSE (+0.015 frente a it6;
+0.018 en pes, +0.031 en v3pes). La caída frente a it4c sigue en todos los escenarios, también en
v2 (−0.054). La causa principal no es el atajo del balón ausente. En IDSSE, it6/it7 predicen menos
rachas que las reales (pred/gt 0.69/0.70 frente a 0.81 de it4c): funden posesiones cortas. Hipótesis
(sin probar): las etiquetas de estilo reto, con menos cambios cortos de posesión, arrastran también
a la cabeza DFL de la red.

## it8: ensamble it4c + it7 (prerregistro, 2026-11-12)

Sin entrenar nada. P(H/A/D) de it8 = 0.5·P(it4c) + 0.5·P(it7), frame a frame, antes de Viterbi.
El peso 0.5 es fijo (no se ajusta). Los costes de Viterbi se ajustan como siempre, con CV anidada.
Se usan las predicciones fuera de muestra que ya existen: it4c en IDSSE (leave-one-match-out) y en
estilo reto (zero-shot, `runs/it4c_x`); it7 con `--kfold 6`.

Regla (la misma): se acepta si `ZX − 0.616 ≥ +0.02` **y** gana a it4c en ≥ 13/23 **y**
`SI ≥ 0.633`. Es la tercera variante evaluada con este criterio sobre los mismos datos (it6,
it7, it8): un aprobado por poco se toma con cautela.

### Resultado de it8

`scripts/hibrido/ensamblar.py` → `runs/it8` (102 pares: IDSSE ×4 escenarios, estilo reto v2).

| modelo | ZX | Metrica | SkillCorner | ganados a it4c | SI (IDSSE) | SI v2 / pes / v3 / v3pes | pred/gt SI |
|---|---:|---:|---:|---:|---:|---|---:|
| it4c | 0.616 | 0.705 | 0.603 | — | 0.663 | 0.725 / 0.616 / 0.701 / 0.609 | 0.81 |
| it7 | 0.678 | 0.698 | 0.675 | 21/23 | 0.589 | 0.671 / 0.526 / 0.637 / 0.521 | 0.70 |
| it8 | **0.686** | 0.729 | 0.679 | **23/23** | **0.660** | 0.728 / 0.611 / 0.700 / 0.603 | 0.78 |

**Decisión según la regla fijada: it8 se acepta** (+0.070 ZX, 23/23, SI 0.660 ≥ 0.633). Pasa a
ser el modelo vigente. Costes de la entrega: los del ajuste conjunto en IDSSE (`it8_D4.json`),
como en it4c: cambio 1, a_muerto 2, desde_muerto 2, sesgo_d 0, temperatura 1.5.

Entrega: `modelos/hibrido/it8.json` (componentes `it4c.pt` e `it7.pt`, este con la bandera de
convención Metrica). `scripts/generar_entrega.py hibrido_it8` → `salidas/hibrido_it8.csv`, válido,
sha256 `e435dbfc6e332049d54bc26c5558aefe30d86fae9c3fac18f365b85c108fa1a0`.
Comprobación única de alineación con el GT público: spellF1 0.533, macroF1 0.808 (it4c:
0.533 / 0.862). Con 21 tramos públicos, la diferencia de macroF1 está dentro del ruido y no se
usa para elegir. Coincide con it4c en el 95.6 % de los frames.

## it9: hill-climbing hacia la entrega (prerregistro, 2026-11-13 01:15)

Plan completo en `docs/PLAN-IT9.md`. Lo vigente es it8 (CSV sha256 `e435dbfc…a1a0`), que queda como
red de seguridad.

**Caja fuerte** (`i % 3 == 2` por fuente sobre ids ordenados; no se miran sus cifras hasta el corte
del lunes 16 a las 12:00): `idsse:J03WOH`, `idsse:J03WQQ`, `metrica:SG3`, `skillcorner:1899585`,
`skillcorner:1953632`, `skillcorner:1996435`, `skillcorner:2006363`, `skillcorner:2010085`,
`skillcorner:2015213`. Dev: los 21 restantes (16 de estilo reto y 5 de IDSSE).

**Métricas.**
- ZX-dev: CV anidada de spellF1 en los 16 partidos de estilo reto de dev, v2, semillas 0 y 1.
- SI-M: IDSSE dev con GT en convención Metrica (`--convencion metrica --minimo_s 2`), media de
  v2/pes/v3/v3pes, como no-inferioridad.
- SI-DFL: solo se reporta.

**E1, costes coherentes con la convención.** Se comparan, con el ensamble it8:
- (a) los costes de la entrega actual, ajustados en IDSSE-DFL (1/2/2/0/1,5);
- (b) los costes ajustados por CV anidada en ZX-dev.

Regla: se adopta (b) si ΔZX-dev ≥ +0,02 frente a (a) y SI-M(b) ≥ SI-M(a) − 0,02. Si se adopta, los
costes de la entrega son los del ajuste con todos los partidos de estilo dev.

**E2, ensamble de 5 semillas por familia.** Jobs 48989929 (it4c_r1–r4) y 48989930 (it7_r1–r4),
`--semilla_red k`. Mezcla: 0,5·media(P de it4c_r0..r4) + 0,5·media(P de it7_r0..r4).
Regla: ΔZX-dev ≥ +0,01 frente al vigente, gana en ≥ 13 de 16 partidos dev y SI-M ≥ SI-M(vigente) − 0,01.

### Hallazgos previos a E1–E4 (2026-11-13)

**E0, diagnóstico** (`scripts/hibrido/diagnostico.py`, it8 con los costes de la entrega, dev). Pierde
sobre todo por equipo y bordes de traspaso, no por vivo/muerto. Oráculos de spellF1:

| | base | vivo/muerto real | equipo real |
|---|---:|---:|---:|
| estilo dev | 0.646 | 0.703 | 0.879 |
| IDSSE dev (GT-M) | 0.672 | 0.694 | 0.951 |

Las posesiones de menos de 3 s son la clase débil (39 % ok en estilo). La duración mínima (E3) puede
dañarlas: se evalúa, pero no se espera ganancia.

**Puerta de E4** (`scripts/hibrido/puerta_tags.py`, sin GT; pasos con balón visible y
max P(H,A) ≥ 0,9 de it8 en BAR-ATH):
- el `Team` de los códigos de ataque coincide con el equipo predicho en el 79,5 % de los pasos;
- en los `DEFENDING*` coincide en el 9,3 %, así que su `Team` es el que defiende y el poseedor es el
  rival, como asume `sesgo_tags` por defecto;
- con esa lectura, la concordancia del equipo poseedor es del 82,6 % ≥ 80 %: la puerta pasa, por
  poco, y es coherente con el volteo nominal ε = 0,2 del simulador.

**E4, regla** (fijada antes de evaluar). `afinar.py --tags nominal`: α, β y γ ∈ {0, 0,5, 1, 2} se
ajustan con los costes por CV anidada en ZX-dev, con tags simulados de ruido nominal (jitter 2 s,
volteo 0,2). La validación se puntúa con:
- tags pesimistas (jitter 3 s, volteo 0,3, cobertura 0,8): `--tags_eval pesimista`;
- tags basura (volteo 0,5): `--tags_eval basura`.

Se acepta si, frente a la misma CV sin tags (E1b), ΔZX-dev(pesimista) ≥ +0,02 con ≥ 13/16 partidos
**y** ΔZX-dev(basura) ≥ −0,01. Solo hay tags simulables en SkillCorner (fases y saques) y en Metrica
(saques); en IDSSE no hay eventos, así que no hay condición de IDSSE.

**E3, regla** (fijada antes de evaluar). `afinar.py --duraciones`: `min_pos` ∈ {0, 2, 3, 5, 8, 10, 15}
y `min_d` ∈ {0, 3, 5, 10, 15, 20, 25} pasos, que se ajustan con los costes por CV anidada en ZX-dev.
Se acepta si ΔZX-dev ≥ +0,01 frente a E1b con ≥ 13/16 partidos, pred/gt ≥ 0,80 y SI-M no inferior
(≥ SI-M(E1b) − 0,01, con los costes de todo el estilo dev).

**E7, IDSSE con convención Metrica (regla fijada antes de entrenar).** Job 48990580: it4m_r0–r2, la
configuración de it4c con `--convencion metrica --minimo_s 2`. it4a no sirve de precedente: sus
`args.json` dicen `convencion dfl`, y su Z se midió con el bug del parser de Metrica.
Comparación del mismo tamaño: 0,5·P(it7) + 0,5·P(it4m_r0) frente a it8m (0,5·P(it7) + 0,5·P(it4c)),
con costes por CV anidada.
Se acepta si ΔZX-dev ≥ +0,01 con ≥ 13/16 partidos y SI-M ≥ SI-M(E1b) − 0,01. Si se acepta, la familia
it4m sustituye a it4c en el ensamble de E2.

### Resultado de E1 (2026-11-13 02:40)

Ensamble it8 con it7 con bandera 1 en IDSSE (`runs/it8m`). CV anidada en ZX-dev; SI-M con los
costes fijos indicados.

| costes | ZX-dev | ganados | pred/gt ZX | SI-M | SI-M v2 / pes / v3 / v3pes | pred/gt SI-M |
|---|---:|---:|---:|---:|---|---:|
| (a) entrega it8 (IDSSE-DFL: 1/2/2/0/1,5) | 0.646 | — | 1.17 | 0.680 | 0.720 / 0.652 / 0.708 / 0.639 | 1.10 |
| (b) CV en estilo dev (4/12/8/0,25/1,0) | **0.688** | **14/16** | 0.84 | **0.712** | 0.771 / 0.662 / 0.754 / 0.660 | 0.87 |

**E1 se acepta** (+0,043 ≥ +0,02, y SI-M sube). Los costes de la entrega pasan a cambio 4,
a_muerto 12, desde_muerto 8, sesgo_d 0,25 y temperatura 1,0: `modelos/hibrido/it9a.json`. it9a es el
nuevo vigente y la base de E2–E7 (ZX-dev 0,688, SI-M 0,712).

### Resultado de E4 (2026-11-13 14:20)

Sesgo del Smart Tagging en Viterbi sobre `runs/it8m`. Pesos ajustados con ruido nominal y evaluados
con ruido pesimista (jitter 3 s, volteo 0,3, cobertura 0,8). Pesos finales: α 0,5, β 0, γ 1.

| | ZX-dev | ganados frente a E1b |
|---|---:|---:|
| E1b (sin tags) | 0.688 | — |
| E4 pesimista | 0.690 | 8/16 |

**E4 se rechaza:** +0,002, lejos de +0,02 con 13/16. E4b (basura) no se corrió, porque ya no cambia
la decisión. Las cadenas de E2/E3/E7 se cortaron a las 03:00 al cerrarse la sesión del agente
(`setsid` no saca el proceso del cgroup). Se relanzaron a las 14:19 como unidad `systemd-run --user`.

### Resultado de E2 (2026-11-13 15:30)

Ensamble de 10 redes (`runs/it9e2`): it7 e it4c, cada una con 5 semillas y peso 0,1. CV anidada en
ZX-dev; SI-M con los costes de ZX-dev (cambio 2, a_muerto 8, desde_muerto 8, sesgo_d 0, T 0,75).

| | ZX-dev | ganados | pred/gt ZX | SI-M | SI-M v2 / pes / v3 / v3pes |
|---|---:|---:|---:|---:|---|
| E1b (it9a) | 0.688 | — | 0.84 | 0.712 | 0.771 / 0.662 / 0.754 / 0.660 |
| E2 | 0.694 | 9/16 | 0.87 | 0.729 | 0.776 / 0.688 / 0.759 / 0.692 |

**E2 se rechaza** según la regla prerregistrada: +0,006 < +0,01 y 9/16 < 13/16. No es peor:
SI-M sube +0,017 (4/5 partidos) y el aumento se concentra en los escenarios pesimistas (+0,026 y
+0,032). No se cambia la regla a posteriori: it9a sigue vigente. E7 se compara contra E1b, como
estaba previsto.

### Resultado de E7 (2026-11-13 15:57)

Ensamble it7 (bandera 1 en IDSSE) + it4m_r0, con 0,5 cada uno (`runs/it9e7`). Costes de ZX-dev:
cambio 2, a_muerto 16, desde_muerto 6, sesgo_d 0,25 y T 0,75.

| | ZX-dev | ganados | pred/gt ZX | SI-M | SI-M v2 / pes / v3 / v3pes |
|---|---:|---:|---:|---:|---|
| E1b (it9a) | 0.688 | — | 0.84 | 0.712 | 0.771 / 0.662 / 0.754 / 0.660 |
| E7 | 0.669 | 2/16 | 0.83 | 0.723 | 0.784 / 0.675 / 0.760 / 0.672 |

**E7 se rechaza:** −0,019 y 2/16 en ZX-dev. Entrenar it4 en IDSSE con la convención Metrica mejora
IDSSE, pero empeora el estilo del reto. it4c se queda en el ensamble.

### Resultado de E3 (2026-11-13 16:17)

Duración mínima en Viterbi sobre `runs/it8m`, con rejillas `min_pos` {0, 2, 3, 5, 8, 10, 15} y `min_d`
{0, 3, 5, 10, 15, 20, 25} pasos. Con todos los partidos la CV elige min_pos 5 (1 s) y min_d 0, y el
resto de costes igual que E1b. Los folds no se ponen de acuerdo: tres eligen (5, 0), dos (15, 15–20)
y uno (0, 0).

| | ZX-dev | ganados | pred/gt ZX | SI-M | ganados SI-M |
|---|---:|---:|---:|---:|---:|
| E1b (it9a) | 0.688 | — | 0.84 | 0.712 | — |
| E3 | 0.689 | 6/16 | 0.87 | 0.712 | 2/5 |

**E3 se rechaza:** +0,0005 y 6/16. La duración mínima no aporta sobre los costes de E1.

### Estado tras it9 (2026-11-13 16:20)

it9a sigue vigente (`modelos/hibrido/it9a.json`): ZX-dev 0,688, SI-M 0,712. E2, E3, E4 y E7 se
rechazan con sus reglas prerregistradas. E2 es la única variante que no empeora en nada (+0,006 en
ZX-dev, +0,017 en SI-M). Queda como alternativa robusta, aunque no pasó la regla.

### it9b: regla dura antes del saque de la segunda parte (2026-11-13 16:40)

El control automático del CSV (PLAN-IT9 §2: «ninguna posesión antes del saque de la segunda parte»)
falla con it9a: el 61 % de los 2.934 frames evaluables entre el 110.808 y el 113.742 salen como
posesión. FAQ.md publica que todos son D. Es la regla dura prerregistrada en E4. Viene de la
documentación pública, no del GT, y no tiene parámetros.

`modelos/hibrido/it9b.json` = it9a + `d_antes_saque: [[110808, 113742]]`. `seccion_con_saques` pone
esos frames como no evaluables antes de Viterbi, así que salen D y la decodificación arranca en el
saque (`tests/test_saque.py`).

- `salidas/hibrido_it9b.csv`: sha256 `97b40a03…1156`, válido, 262 posesiones (it9a: 266).
- Respecto a it9a cambian 1.799 frames, todos dentro de la ventana.
- Comprobación pública única: 0,5455, igual que it9a, porque los 5 minutos públicos no incluyen la
  ventana.
- Controles: 4,5 posesiones por minuto vivo, 18 % de menos de 3 s, D 0,45 en frames evaluables y
  ninguna posesión antes del saque. Pasa los cuatro.

**it9b es el candidato de entrega.**

## E5: escenario v4, simulación más cercana al ATD (prerregistro, 2026-11-13 17:30)

**Cambio.** El escenario `v4` (`mtah/escenarios.py`) es v3 más lo siguiente:
- `aleatorizar`: los parámetros calibrados con agregados NO publicados del ATD se sortean por
  semilla en rangos que contienen el valor calibrado (`degradar.RANGOS`): mediana y sigma de los
  huecos de balón, mediana y sigma de los fragmentos, hueco de jugador y ruido. Fijo queda solo lo
  publicado en DATA.md: 40,7 % sin balón y 46 slots.
- Defectos que describe DATA.md, con tasas fijadas a priori: duplicados 0,5/min (a veces más de 22
  visibles), huecos más probables con el balón limpio a > 15 m/s (peso 2), balón falso 0,3/min y
  equipo erróneo al recuperar una pista (3 % de los fragmentos).
- v2/v3/pes/v3pes no cambian ni un bit (`tests/test_degradar_v4.py`).
- En IDSSE, v4 da 40,5–41,2 % sin balón, 46 slots y 20,7–21,6 visibles (ATD: 40,7 %, 46 y 20,5).

**Brazos (solo IDSSE, misma receta salvo la degradación de entrenamiento).**
- Base: it4c, it4c_r1 e it4c_r2 (`--deg_train v3`; ya entrenados).
- v4: it4v_r0, it4v_r1 e it4v_r2 (`--deg_train v4 --semilla_red 0/1/2`).
- Receta común: `--final --semillas 6 --epocas 30 --escenarios_eval v2+pes+v3+v3pes --peso_borde 3 --mezcla`.

**Métricas.**
- Primaria: ZX-dev zero-shot de cada modelo (`final.pt` sobre Metrica y SkillCorner dev, escenario
  v2). Se calcula con `afinar.py <run>_x --fuente_datos estilo --kfold 6 --dev --escenarios v2` y
  se promedia por brazo.
- No inferioridad: SI-DFL (OOF LOGO de IDSSE, `afinar.py <run> --dev --escenarios v2,pes,v3,v3pes`),
  promediada por brazo.

**Regla.** v4 pasa si Δ ZX-dev ≥ max(0,02, 2·σ), con σ la mayor de las desviaciones típicas entre
semillas (ddof 1) de los dos brazos, y además Δ SI-DFL ≥ −0,02. Si pasa, el job C reentrena it7 con
v4 y el ensamble it7v + it4v se compara con it9b con la regla de E2 (ΔZX-dev ≥ +0,01, ≥ 13/16 y
SI-M ≥ −0,01). Si no pasa, it9b se queda.

### Resultado de E5 (2026-11-13 22:06)

`scripts/hibrido/decidir_e5.py` (jobs MN5 49016678–80, 13 min cada uno):

| brazo | ZX-dev por semilla | media | SI-DFL por semilla | media |
|---|---|---:|---|---:|
| base (v3) | 0.617 / 0.603 / 0.614 | 0.611 | 0.657 / 0.675 / 0.675 | 0.669 |
| v4 | 0.629 / 0.606 / 0.612 | 0.616 | 0.657 / 0.680 / 0.671 | 0.669 |

Δ ZX-dev +0,004, con umbral max(0,02, 2σ) = 0,024 (σ = 0,012) y 10/16 partidos ganados. Δ SI-DFL
+0,0004. **E5 no pasa:** v4 no empeora, pero la mejora queda dentro del ruido entre semillas. No se
lanza el job C. **it9b sigue como candidato de entrega.**

Como efecto secundario, quitar la calibración con agregados no publicados del ATD (rangos
aleatorios en `v4`) no cambia el rendimiento. Para el documento técnico: la ganancia de it8/it9 no
depende de esa calibración.

## Congelación (2026-11-16 10:50)

**Caja fuerte** (una sola vez; `runs/it8m`, `--solo_caja --costes_fijos`):

| | estilo (6 SkillCorner + SG3) | ganados | SI-M (J03WOH, J03WQQ) | ganados |
|---|---:|---:|---:|---:|
| it8 tal como se entregó (1/2/2/0/1,5) | 0.627 | — | 0.679 | — |
| it9b (4/12/8/0,25/1,0) | **0.677** | **6/7** | **0.722** | 2/2 |

Regla del plan: Δ ≥ 0 y ≥ 4/7 de estilo. **Pasa: it9b es el modelo final.**

**Determinismo.** El venv del worktree se borró durante el fin de semana y se recreó desde
`requirements-equipo.txt`, con torch 2.15.0+cpu ya fijado. `generar_entrega.py hibrido_it9b`
reproduce el CSV bit a bit: sha256 `97b40a03…1156`, igual que el del 13-nov.

## E8: TTA (prerregistro, 2026-11-16 11:00)

Sin entrenar (MN5 parado desde el 14-nov; todo en CPU local, una cadena a la vez).

**Cambio.** Cada componente de it9b predice con la media de P(H, A, D) sobre las 8
transformaciones que ya usaba la aumentación de entrenamiento: reflejo x, reflejo y y cambio de
equipos, deshaciendo H↔A (`modelo.predecir_tta`, `tests/test_tta.py`). El ensamble sigue siendo 0,5 / 0,5 y
los costes se reajustan con la misma CV anidada que E1b.

**OOF.**
- it4c: IDSSE leave-one-match-out con `fold_<partido>.pt` y estilo zero-shot con `final.pt`.
- it7: `fold_k*.pt`, con bandera 1 en IDSSE.
- Comprobado sin TTA: la ruta local reproduce los OOF de MN5 (diferencia máxima de P 0,0005).

**Regla** (la de E2, frente a E1b): ΔZX-dev ≥ +0,01, ≥ 13/16 partidos y ΔSI-M ≥ −0,01. Como la caja
fuerte ya se usó, si E8 pasa en dev se mira la caja por **segunda y última vez**: tiene que dar
Δ ≥ 0 frente a it9b con ≥ 4/7 en estilo. Solo entonces sustituye a it9b, como it9c
(`"tta": true`).

### Resultado de E8 (2026-11-16 14:15)

`runs/it9e8` (102 pares). Costes de ZX-dev: cambio 4, a_muerto 24, desde_muerto 8, sesgo_d 0,25 y
T 1,5.

| | ZX-dev | ganados | pred/gt ZX | SI-M | ganados SI-M |
|---|---:|---:|---:|---:|---:|
| E1b (it9b) | 0.688 | — | 0.84 | 0.712 | — |
| E8 (TTA ×8) | 0.686 | 7/16 | 0.92 | 0.708 | 1/5 |

**E8 se rechaza:** −0,003 en ZX-dev y −0,003 en SI-M. No hace falta la segunda mirada a la caja
fuerte. El código de TTA se queda, apagado por defecto (`"tta"` no aparece en it9b.json).
**it9b es la entrega final.**
