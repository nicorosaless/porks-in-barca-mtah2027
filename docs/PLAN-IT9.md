# Plan it9: cierre del reto (13–17 nov 2026)

Autor: Fable 5.2 (esfuerzo alto), 2026-11-13. Solo leyó código, documentación y
`salidas/hibrido/*.json`: nada de `data/` ni CSV de BAR-ATH. Las ganancias son estimaciones suyas,
no medidas.

Restricciones (correo oficial del 3-nov): entrenar y ajustar solo con datos externos, y nada de
pseudoetiquetas sobre BAR-ATH. Solo pasan los 20 mejores equipos. La entrega cierra el 17-nov a
las 23:59 CET.

## 1. Diagnóstico

El scorer empareja posesiones 1 a 1, del mismo equipo, por IoU con umbrales de 0,3 a 0,7. D no
puntúa: solo cuenta por los bordes que crea.

| error | coste | palanca |
|---|---|---|
| Fundir dos posesiones (D corto o racha rival omitidos) | 1 FN en los 5 umbrales; la superviviente cae bajo 0,5–0,7 | costes, duración mínima de D, saques del Smart Tagging |
| Partir (racha o D espurios) | 1–2 FP | costes, duración mínima |
| Equipo equivocado | FN + FP en todos los umbrales | ensamble, fases con equipo del Smart Tagging |
| Borde desplazado | IoU = (L−e)/(L+e): una posesión de 3 s tolera 0,5 s a IoU 0,7 | pérdida en bordes, latencia por tipo |

Hallazgos:

1. **Los costes entregados no son los validados** (comprobado 2026-11-13).
   - it8 se aceptó por ZX con los costes ajustados en estilo reto: cambio 4, a_muerto 12, desde_muerto 4, sesgo_d 0,25, T 1,0 (`it8_ZX.json`).
   - `modelos/hibrido/it8.json` lleva los de IDSSE/DFL: 1/2/2/0/1,5. Están ajustados a posesión por toque, y el reto pide posesión controlada.
   - Nunca se midió ZX con los costes de IDSSE.
2. **La combinación que se usa en BAR-ATH no está validada.** BAR-ATH tiene campo completo (~20 jugadores visibles) y se predice con la bandera 1. Esa bandera solo se evaluó en SkillCorner (unos 9 jugadores visibles) y en Metrica.
3. **El modelo funde posesiones en todos los dominios.** pred/gt: 0,85 en ZX, 0,75 en Metrica y 0,78 en SI.
4. **El 0,53 del GT público no mide la brecha.** Son 17 posesiones, así que cada una vale unos 6 puntos. Los baselines bajan entre 0,08 y 0,20 del público al partido completo.
5. **El balón falta en bloques largos.** Ahí solo quedan como pista los jugadores y el Smart Tagging.

## 2. Experimentos, por prioridad

| # | cambio | ganancia est. | coste | riesgo | regla prerregistrada |
|---|---|---|---|---|---|
| E0 | `scripts/hibrido/diagnostico.py` sobre los OOF de it8. Cada posesión GT queda como ok, borde, fundida, partida, equipo o perdida, por duración y por fracción de balón visible. Se calculan oráculos (vivo/muerto real, equipo real). | ordena E3–E5 | 3 h local | nulo | Se descarta toda palanca cuyo oráculo dé < +0,03. |
| E1 | Costes coherentes con la convención: ajuste conjunto estilo + IDSSE con GT-M (`--convencion metrica --minimo_s 3`) y peso igual por grupo. Matriz de ZX y SI-M con unos costes y con otros. Necesita el OOF de it7 en IDSSE con bandera 1. | +0,02–0,06 | 2 h local, 5 min MN5 | bajo | Costes conjuntos si ΔZX-dev ≥ +0,02 frente a los de IDSSE y SI-M no baja más de 0,02. |
| E2 | Ensamble de semillas (`--semilla_red`) y TTA: reflejos X/Y e intercambio de equipos. Job A: 4 semillas más de it4c y 4 de it7. Pesos fijos de 0,5 por familia, uniformes dentro. | +0,01–0,02 y menos varianza | 2 h de código, ~1 h 45 de GPU | bajo | ΔZX-dev ≥ +0,01, ≥ 13/16 y SI-M ≥ −0,01. |
| E3 | Duración mínima en Viterbi por expansión de estados (`min_pos`, `min_d`). Rejillas {0, 3, 5, 8, 10, 15} y {0, 5, 10, 15, 25} pasos. HSMM solo si E0 lo justifica. | +0,01–0,03 | 4–5 h local | medio | ΔZX-dev ≥ +0,01, ≥ 13/16, pred/gt ≥ 0,80 y SI-M no inferior. |
| E4 | Smart Tagging en el decodificador, sin entrenar. Detalle abajo. | +0,02–0,05 | 1–1,5 días | alto | Abajo. |
| E5 | Simulación más cercana al ATD (escenario v4) y resolver la zona gris de calibración. Detalle abajo. | 0–0,04 | 4 h de código y jobs B y C | medio | Abajo. |
| E6 | Diversidad de arquitectura en el job B: `--d 96 --dt 192`, `--largo 1024`, `--aux 0.5`. | 0–0,01 | GPU | bajo | La de E2. |

**E4, Smart Tagging**

- **Evidencia.** Nuevo `mtah/fuentes/smart_tagging.py`. Por paso da el equipo de la fase (códigos de ataque: `Team`; `DEFENDING*`: el rival) y la ventana de reanudación (los saques de banda llevan `N/A`).
- **Potenciales en Viterbi.** +α al equipo de la fase, −γ a D dentro de una fase, +β a D en los 2 s previos a un saque y al equipo en los 2 s posteriores. Rejilla {0, 0,5, 1, 2}, con tope de 2 nats.
- **Reglas duras a priori.** D antes de cada `Kick Off (Start)` y entre `GOALS` y `Kick Off (After Goal)`. FAQ.md publica que los primeros 2.934 frames evaluables de la segunda parte son D.
- **Simulación de etiquetas en datos externos.**
  - SkillCorner: `phases_of_play` y `dynamic_events`.
  - IDSSE: eventos DFL.
  - Metrica: `SET PIECE`.
  - Ruido fijado a priori contra la fuga: jitter de 1–2 s, cobertura recortada a la del fichero real, volteo de equipo ε ∈ {0,1, 0,2, 0,3} y recall por tipo de saque.
- **Regla.**
  - Pesos por minimax entre el ruido nominal y el pesimista, en CV anidada.
  - Se acepta si Δ ≥ +0,02 en el escenario pesimista con ≥ 13/16.
  - Con etiquetas basura (ε = 0,5) tiene que perder ≤ 0,01.
  - Los saques deben dar ≥ 0 en IDSSE.
- **Puerta en BAR-ATH, sin etiquetas y con umbral fijado ya.** La concordancia entre el equipo del Smart Tagging y el de it8 (balón visible, P ≥ 0,9) debe ser ≥ 80 %. Si no, solo entran las reglas duras.

**E5, simulación → ATD**

- Como valores fijos quedan solo las cifras publicadas en DATA.md: 40,7 % sin balón, 46 slots y 20,5 jugadores visibles. Lo calibrado con agregados no publicados pasa a rangos aleatorios por semilla.
- Defectos nuevos que describe DATA.md: balón falso, jugadores duplicados, equipo erróneo al readquirir y huecos con balón rápido o aéreo (escenario `v4`).
- Proxy de «real»: zero-shot de un modelo solo-IDSSE entrenado con v4, medido en SkillCorner-dev; 3 semillas frente a 3.
- Regla: Δ ≥ max(0,02, 2σ_semilla) y SI ≥ −0,02. Si pasa, el job C reentrena las dos familias.

**Fuera del plan:** balón desde el vídeo, broadcast, PFF y pseudoetiquetas (prohibidas).

**Control automático del CSV final**, con rangos externos fijados ya:
- 3,5–6,5 posesiones por minuto vivo;
- 10–35 % de posesiones de menos de 3 s;
- D entre 0,25 y 0,55;
- ninguna posesión antes del saque de la segunda parte.

## 3. Hill-climbing sin sobreajustar

- **Caja fuerte.** Por fuente e id ordenado, los partidos con `i % 3 == 2`: 6 de SkillCorner, 1 de Metrica y 2 de IDSSE. Sus cifras no se miran hasta el corte. El resto es dev.
- **Métrica primaria.** ZX-dev, emparejada por partido. SI-M como no-inferioridad.
- **Comparaciones.** Como máximo 6 confirmatorias, escritas aquí o en HIBRIDO.md antes de correr. Cada una exige efecto mínimo y test de signos ≥ 13/16.
- **Orden voraz y fijo:** E1 → E2 → E3 → E4 → E5, cada una contra el mejor vigente.
- **Semillas.** 5 por familia. Las variantes de entrenamiento se comparan con ensambles del mismo tamaño.
- **Caja fuerte, una sola vez.** Candidato final frente a it8: Δ medio ≥ 0 y ≥ 4/7 de estilo. Si falla, se entrega el último candidato que la pase, o it8.
- **GT público.** Una vez, tras elegir. Si sale < 0,43, se busca un error de alineación.

## 4. Calendario

| día | modelado | cierre |
|---|---|---|
| Vie 13 | Caja fuerte. Código de E2 y job A en cola. E1 en local. Empezar E0. | Etiqueta y sha de it8. Pregunta a los organizadores. |
| Sáb 14 | E0. E3. Evaluar el job A. Código de E5 y job B. Parser y simulador de E4. | Subir it8 al formulario con repo y documento mínimos. Determinismo. |
| Dom 15 | E4 y su puerta. Evaluar el job B; job C si v4 pasa. 20:00: no se lanza nada nuevo. | Documento, licencias y citas. |
| Lun 16 | 12:00: congelación. Caja fuerte, `final.json`, CSV y comprobación pública. | Clon limpio con sha256 idéntico. Actualizar el formulario antes de las 20:00. |
| Mar 17 | Nada. | Colchón; cierre a las 18:00. |
