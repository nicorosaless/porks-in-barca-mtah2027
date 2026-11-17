# Estado del arte (research del 2026-11-04)

Resumen de tres búsquedas hechas con subagentes. Las marcadas *(sin verificar)* se basan en
resúmenes de búsqueda, no en el texto original; hay que leerlas antes de construir sobre ellas.

## 1. Posesión desde tracking

| trabajo | idea | por qué nos importa |
|---|---|---|
| **PathCRF**, Kim et al., KDD 2026 ([arXiv 2602.12080](https://arxiv.org/abs/2602.12080), [código](https://github.com/hyunsungkim-ds/pathcrf)) | Infiere el poseedor sin balón: grafo de jugadores + CRF + Viterbi | Pensado justo para cuando falta el balón. Licencia del repo sin comprobar |
| **Vidal-Codina et al. 2022** ([arXiv 2202.00804](https://arxiv.org/abs/2202.00804)) | Posesión por zonas de distancia y frames de control; eventos por reglas | La mejor plantilla de reglas para posesión y balón parado |
| **Link & Hoernig 2017**, PLOS ONE ([PMC5503225](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5503225/)) | Golpeos por picos de aceleración del balón; posesión de equipo = cadena de posesiones individuales | El principio anti-fragmentación que pide spellF1 |
| **Li & Link 2026** ([arXiv 2606.09289](https://arxiv.org/abs/2606.09289)) | Atención temporal en grafos + Transformer sobre TRACAB | Reporta tIoU además de F1 por frame; etiquetas de fases, no de posesión |
| **databallpy** ([GitHub](https://github.com/Alek050/databallpy)) | Posesión individual al estilo Vidal-Codina, integrada con kloppy | Baseline de reglas mantenido y reutilizable |

## 2. Balón muerto y balón ausente

| trabajo | idea | por qué nos importa |
|---|---|---|
| **Kim et al., KDD 2023** ([arXiv 2306.08206](https://arxiv.org/abs/2306.08206)) | Set Transformer + Bi-LSTM jerárquico: infiere la trayectoria del balón desde los jugadores | Rellenar el 41 % sin balón. Con un 20 % de balón observado, 1,41 m de error *(sin verificar)*. Sin código |
| **TranSPORTmer**, ACCV 2024 ([paper](https://openaccess.thecvf.com/content/ACCV2024/papers/Capellera_TranSPORTmer_A_Holistic_Approach_to_Trajectory_Understanding_in_Multi-Agent_Sports_ACCV_2024_paper.pdf)) | Transformer enmascarado: imputación e inferencia de balón | Candidato para la línea C. Código no encontrado |
| **SoccerNet 2025** ([arXiv 2508.19182](https://arxiv.org/abs/2508.19182)) | Ganadores de Ball Action Spotting (T-DEED reconstruido) y GSR | Referencia de vídeo. Datos bajo NDA; T-DEED es GPL-3.0 |
| **SoccerTrack v2** ([arXiv 2508.01802](https://arxiv.org/abs/2508.01802)) | 10 partidos con cámara fija panorámica 4K, balón y acciones | Lo más parecido a nuestro vídeo táctico, para un detector de balón con datos externos |
| Roboflow football-ball-detection (CC BY 4.0) | ~1.200 imágenes con el balón anotado | Datos externos para afinar un detector |

## 3. Datasets externos

| dataset | partidos | fps | vivo/muerto | posesión | acceso | licencia |
|---|---|---|---|---|---|---|
| **IDSSE (DFL)** | 7 | 25 | sí, `BallStatus` por frame | sí, `BallPossession` por frame | [HF](https://huggingface.co/datasets/pysport/idsse-data) | CC BY 4.0, **aprobado en la #16** |
| Metrica sample-data | 3 | 25 *(sin verificar)* | derivable de eventos | derivable de eventos | [GitHub](https://github.com/metrica-sports/sample-data) | sin licencia formal |
| PFF WC 2022 | 64 | ~30 | campo `ball_state` | campo `ball_owning_team_id` | formulario | sin comprobar |
| SkillCorner open data | 10 | 10 | sin verificar | sin verificar | GitHub | MIT |

Cita obligatoria de IDSSE: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). *An integrated
dataset of spatiotemporal and event data in elite soccer.* Scientific Data 12, 195.
https://doi.org/10.1038/s41597-025-04505-y

## Nota sobre otros equipos

La búsqueda encontró repos públicos de otros participantes de este mismo reto. No los usamos como
fuente de código: es su entrega, y nuestro documento técnico tiene que explicar un método nuestro.
Las ideas genéricas que aparecen (augmentación con huecos de balón, Viterbi sobre features por
frame) ya están en la literatura citada arriba.

## Recomendaciones que salen del research

1. **Decodificación temporal con coste de cambio** (Viterbi/HMM/CRF de 3 estados H/A/D con
   duración mínima). Es lo que más ataca la fragmentación, que es lo que castiga spellF1.
2. **Poseedor sin balón** al estilo PathCRF o Kim 2023, entrenado en IDSSE/PFF degradados, como
   entrada a la máquina de estados en los huecos.
3. **Reglas de balón muerto** al estilo Vidal-Codina más el Smart Tagging (set pieces) como
   evidencia, no como verdad.
