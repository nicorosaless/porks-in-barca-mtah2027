# Plan del equipo — More than a Hack 2027

Borrador del 2026-11-04 para discutir en equipo. Las decisiones marcadas como **propuesta** están
abiertas; el resto ya está fijado por las reglas o por las respuestas de los organizadores.

## Objetivo

Ganar el ranking de **spellF1** (posesiones emparejadas por IoU temporal), con macroF1 como
desempate. Suelo publicado: 0,2433. Ruido entre equipos: ±0,04. Entrega el **16-nov** (deadline
oficial 17-nov).

## Restricciones fijas

De RULES.md y de las issues respondidas del repo oficial:

| tema | regla | fuente |
|---|---|---|
| Etiquetas | Nada de etiquetas a mano sobre BAR-ATH, ni para entrenar, validar o elegir parámetros | RULES.md, #2, #10 |
| Entrenar con el partido | **No** se puede entrenar sobre BAR-ATH usando su tracking/homografía como etiquetas (p. ej. un detector de balón) | #2 |
| Datos externos | Sí, en cualquier cantidad, documentando la fuente. IDSSE confirmado explícitamente | #16 |
| Ajustes | Todo hiperparámetro y umbral se elige en IDSSE u otros partidos, nunca en el GT público | #16 |
| Futuro | Se pueden usar frames futuros (post-match, no causal) | #15 |
| Hardware | El tiempo de ejecución no importa; GPU CUDA aceptada en la reproducción | #2 |
| Reproducción | El repo debe regenerar **el mismo** CSV | RULES.md |
| LLM | No pasar datos ni extractos del partido a LLMs alojados | OVERVIEW.md |
| Preguntas | Nuevas dudas por correo: hackathon.talentarena@mobileworldcapital.com | #22 y siguientes |

## Decisiones tomadas

1. **Equipo:** de momento, Nico + agente. El reparto entre los cuatro se decide en la reunión.
2. **Enfoque híbrido (c):** una máquina de estados (vivo/muerto + posesión con histéresis) es la
   columna vertebral. Los modelos aprendidos alimentan sus entradas (balón inferido cuando falta,
   probabilidad de balón muerto).
3. **Validación externa:** IDSSE (7 partidos, flags de balón vivo y posesión por frame) degradado
   para parecerse al ATD. Metrica sample y PFF como apoyo. Ya montado: ver abajo.
4. **MareNostrum:** permitido por la asignatura. Se usa para entrenar con datos externos.
5. **Calendario y cortes:** ver abajo.
6. **Repo de entrega:** repo público nuevo y limpio el 15-nov, con solo la solución.

## Calendario (propuesta)

| fecha | hito | criterio |
|---|---|---|
| 4–6 nov | Base: validación externa, port del baseline, degradación calibrada ✅ | `pytest` verde |
| 4–6 nov | Reunión de equipo: reparto y estrategia | — |
| 7 nov | **Corte 1:** heurística v1 frente al baseline en IDSSE degradado | +0,10 spellF1 o todo el equipo a la máquina de estados |
| 8–12 nov | Modelos en MareNostrum (balón inferido, vivo/muerto) e integración | — |
| 12 nov | **Corte 2:** un modelo entra solo si mejora > 0,04 en validación | — |
| 13–14 nov | Integración final. **Congelación del código el 14** | — |
| 15–16 nov | Repo público, reproducción desde clon limpio (hash idéntico), documento técnico, entrega | sha256 igual |

## Base disponible (rama `T-2611-base`)

| pieza | qué hace |
|---|---|
| `mtah/partido.py` | Representación común `Partido`: balón, slots, equipos, secciones, GT |
| `mtah/fuentes/reto.py` | ATD táctico del reto → `Partido` (cache en `data/cache`) |
| `mtah/fuentes/idsse.py` | IDSSE → `Partido`, con GT H/A/D desde BallStatus/BallPossession |
| `mtah/degradar.py` | Simula el ATD: 40,7 % de balón ausente en bloques largos, 46 slots, fragmentación |
| `mtah/predictores/` | Registro de métodos. Hoy: los 4 baselines oficiales, portados y verificados |
| `mtah/evaluar.py` | Puntúa con el scorer oficial, sin reimplementar métricas |
| `scripts/evaluar_externo.py` | Tabla de spellF1/macroF1 por método en IDSSE limpio y degradado |
| `scripts/generar_entrega.py` | CSV de entrega + validador oficial + sha256 |
| `scripts/calidad_tracking.py` | Compara agregados de calidad ATD frente a IDSSE degradado |

### ¿Sirve IDSSE degradado como validación?

Primera prueba (`scripts/evaluar_externo.py --limpio --semillas 2`, 7 partidos):

| baseline | IDSSE limpio | IDSSE degradado | BAR-ATH completo (oficial) |
|---|---|---|---|
| raw | 0,098 | 0,050 | 0,049 |
| smooth25 | 0,296 | 0,186 | 0,200 |
| smooth50 | 0,308 | **0,199** | **0,243** |
| smooth125 | 0,218 | 0,169 | 0,224 |

En degradado, el orden de los cuatro coincide con el oficial y la escala es parecida. En limpio,
todo sale más alto: validar sin degradar engaña. La dispersión entre partidos es de unas 0,06,
así que hay que comparar con los 7 partidos y varias semillas, no con uno.

El macroF1 en IDSSE (~0,44) es más bajo que en BAR-ATH (0,67) porque el baseline solo dice D
cuando falta el balón, e IDSSE tiene el doble de balón muerto. Otra señal de que el balón muerto
necesita su propia lógica.

## Riesgos conocidos

- **La convención de IDSSE no es la de Metrica.** El GT del reto sale del tracking corregido de
  Metrica; el de IDSSE, de los flags de DFL. Cuándo empieza un balón muerto o cuándo cambia la
  posesión en un duelo puede diferir (ver la issue #31 abierta y la #17). Un método puede ganar en
  IDSSE por imitar a DFL. Mitigación: comprobar en los 5 minutos públicos que el comportamiento no
  se rompe (comprobación, no ajuste) y preferir reglas que se justifiquen con CHALLENGE.md.
- **Proporciones distintas:** en IDSSE (J03WMX) el balón muerto es el 41 % de los frames; en los
  5 minutos públicos del reto, el 20 %. Los criterios de corte deben mirar posesiones, no frames.
- **Dónde falta el balón:** la degradación coloca los huecos al azar. En el ATD real faltan en
  sitios concretos (congestión, balones aéreos, ¿balón parado?). Pendiente de estudiar con
  agregados.

## Preguntas abiertas para la reunión

1. Reparto de las cuatro líneas: A máquina de estados, B balón muerto y Smart Tagging,
   C modelos en MareNostrum, D validación, reproducibilidad y documento.
2. ¿Escribimos a los organizadores para preguntar por el inicio y fin de D (como la #31)?
3. ¿Pedimos acceso a PFF (formulario) para tener volumen de entrenamiento?
4. Nombre del equipo y del repo público.
