# Plan it4: cerrar la brecha IDSSE → BAR-ATH

Plan redactado por Fable 5.1 el 7-nov-2026; lo ejecuta GPT-5.5 (razonamiento medio). Entrega: 17-nov. Congelación de código del equipo: 14-nov (`docs/PLAN.md`).

## 0. Qué hay y qué falla

Lee primero, en este orden: `CHALLENGE.md`, `EVALUATION.md`, `RULES.md`, `DATA.md`, `FAQ.md`, `docs/PLAN.md`, `docs/HIBRIDO.md`, `docs/RESEARCH.md`. Después el código: `mtah/partido.py`, `mtah/degradar.py`, `mtah/fuentes/idsse.py`, `mtah/hibrido/{rasgos,modelo,decodificar}.py`, `scripts/hibrido/{entrenar,afinar,calibrar_balon}.py`, `scripts/hibrido/entrenar_mn5.sbatch`, `mtah/predictores/hibrido.py`, `scripts/evaluar_externo.py`, `tests/`.

Estado del repo: worktree `<repo>`, rama `T-2611-base`, todo sin commit (`mtah/`, `scripts/`, `tests/`, `docs/` y `modelos/` son untracked). `data/` y `salidas/` están en `.gitignore`.

Entorno:

```bash
cd <repo>
export MTAH_EXTERNAL_DIR=/ruta/a/external
PYTHONPATH=. .venv/bin/pytest -q          # hoy: 10 passed. Sin PYTHONPATH falla al importar mtah
```

Ahora mismo `.venv/bin/pytest -q` a secas falla (`ModuleNotFoundError: mtah`). El paso 1 añade un `conftest.py` para que funcione tal como pide la entrega.

Datos externos disponibles: solo IDSSE, 7 partidos (`$MTAH_EXTERNAL_DIR/idsse`, caché en `$MTAH_EXTERNAL_DIR/cache/idsse_*.npz`). No hay Metrica sample ni PFF.

En local hay 16 CPU y 46 GB de RAM, sin GPU, con `torch 2.15.0+cpu` y `kloppy 3.18.0` instalados.

MareNostrum 5:
- Acceso: `ssh marenostrum` (nodo de login, usuario del proyecto), sin contraseña.
- Ya existen `$HOME/mtah2027/{code,external/cache,runs,logs}`, con los runs anteriores `it2, it3a, it3a_pes, it3b, it3c`.
- Un run de 7 folds más el final tardó 756 s (it3c).
- Configuración (cabecera de `scripts/hibrido/entrenar_mn5.sbatch`): qos `acc_debug`, 2 h, 1 GPU.
- En esa qos solo corre un trabajo a la vez. Si aparece `tts-esti` en `squeue -u $USER`, hay que esperar. Nunca hagas `scancel` de nada que no sea `mtah-hibrido`.

### Diagnóstico con números (medidos el 7-nov, solo agregados)

| magnitud | IDSSE (DFL), 7 partidos | it3c en BAR-ATH (predicción, partido completo) | GT público BAR-ATH (DATA.md, 5 min) |
|---|---|---|---|
| fracción D en frames evaluables | 0,41–0,52 | 0,44 / 0,46 (1ª/2ª parte) | 0,20 |
| duración media de un tramo D | 22–28 s (mediana 15–24 s) | 31 s / 23 s | 14,7 s (4 tramos) |
| duración media de posesión | 5,5–9,2 s | 11,4 s / 8,6 s | 14,2 s |
| posesiones < 3 s | 38–50 % | 27 % / 30 % | — |
| posesiones por minuto de balón vivo | 7–11 | — | 4,2 |

Lectura:
- DFL marca un cambio de posesión en cada toque: la mitad de sus posesiones duran menos de 3 s.
- `CHALLENGE.md` dice lo contrario. Un toque sin control, una desviación o un duelo aéreo sin control **no** cambian la posesión. Con el balón disputado o suelto, la posesión sigue con el último equipo que la tuvo.
- La red it3c ha aprendido la convención de DFL: en BAR-ATH produce ~30 % de posesiones de menos de 3 s.
- Con la convención de Metrica, cada una de esas posesiones es un falso positivo **y** además parte una posesión verdadera en dos trozos, cada uno con IoU ≈ 0,5.
- Eso explica el perfil observado: recall alto a IoU 0,3 (0,71) y hundimiento a IoU 0,7 (P/R 0,25/0,29, frente a 0,67/0,62 en IDSSE).

Dónde falta el balón en el ATD, comparado con `J03WMX` degradado (solo agregados del input):
- Se mira el estado del frame anterior al inicio de cada hueco.
- Grupo de control: 5000 frames aleatorios con balón.

| al inicio del hueco | ATD | control ATD | IDSSE deg. v2 | IDSSE deg. pesimista |
|---|---|---|---|---|
| balón dentro de un área | 0,30 | 0,15 | 0,26 | 0,21 |
| balón fuera del campo | 0,15 | 0,04 | **0,53** | 0,17 |
| distancia mediana al jugador más cercano (m) | **1,45** | 1,90 | **2,62** | **2,31** |
| jugadores a < 5 m (media) | 1,60 | 1,40 | 1,25 | 1,31 |
| nº de huecos | 149 | — | 121 | 149 |
| huecos < 1 s / 1–5 s / 5–20 s / > 20 s | 16 / **1** / 88 / 44 | — | 10 / 17 / 52 / 42 | 2 / 42 / 64 / 41 |
| fracción de frames sin balón en huecos > 20 s | 0,67 | — | 0,74 | 0,67 |

Lectura:
- En el ATD el balón se pierde **cerca de jugadores** (duelos, recepciones) y en las áreas.
- En nuestra degradación se pierde al azar y lejos de los jugadores. En v2, además, demasiado a menudo fuera del campo.
- El ATD casi no tiene huecos de 1–5 s: o duran menos de 1 s o 5 s o más. Parece que Metrica interpola los cortos.

## 1. Restricciones duras

1. El GT público de BAR-ATH (5 min) se mira **una sola vez**: al final de it4, con el modelo ya elegido, para comprobar la alineación. Nunca sirve para elegir, ajustar, parar entrenamientos ni fijar umbrales. `scripts/generar_entrega.py` imprime esa cifra: **no lo ejecutes hasta el paso final**.
2. Nada de etiquetar BAR-ATH a mano, ni de reglas o constantes atadas a frames concretos del partido.
3. Nada de `data/` en git ni subido a ningún sitio. `git status` debe seguir mostrando solo código, docs, tests y `modelos/`.
4. A MareNostrum solo van código y datos externos (IDSSE, Metrica sample). La inferencia de BAR-ATH se hace en local.
5. No imprimas filas en bruto del ATD de BAR-ATH en el contexto del agente: solo agregados, esquemas y código. Las filas de IDSSE y de Metrica sample son públicas y sí se pueden ver.
6. Sin commit, push, correo ni issue. El borrador de la pregunta a los organizadores (sección 7) se deja escrito en `docs/PREGUNTA-ORGANIZADORES.md`, sin enviar.
7. No uses `sleep` para esperar: procesos en background y monitores (un bucle `until` con `ssh marenostrum squeue`).
8. No pares procesos ajenos (`tts-esti` ni ningún otro).
9. **El criterio de selección queda fijado antes de mirar resultados** (sección 4) y no se cambia después.

## 2. Palancas, ordenadas por valor esperado

### P1. Convención de posesión de Metrica: segundo dominio y reetiquetado de IDSSE (día 1)

**Por qué.** Según el diagnóstico, DFL etiqueta a nivel de toque, mientras que el reto pide posesión *controlada* (`CHALLENGE.md`, «What does not count as a change of possession»). El modelo reproduce la convención de DFL en BAR-ATH: un 27–30 % de posesiones de menos de 3 s. Es el fallo que más castiga el spellF1 a IoU alto, que es justo lo que nos falta. El reetiquetado solo cambia el 1,7–3,1 % de los frames de IDSSE, pero son precisamente los bordes.

**Qué se hace.**
- (a) Traer **Metrica sample-data**: 3 partidos a 25 fps, campo de 105×68 y coordenadas 0–1 con el origen arriba a la izquierda, el mismo convenio que el ATD. Las etiquetas H/A/D se derivan de sus **eventos**.
  - Es el único dato público con la convención de Metrica.
  - Sirve para fijar el umbral del reetiquetado sin tocar BAR-ATH y para evaluar *zero-shot* la transferencia de convención.
- (b) Reetiquetar IDSSE a «posesión controlada» y entrenar y validar con ese GT.
  - Regla: una posesión de menos de `T` segundos, precedida por una posesión viva del otro equipo, se absorbe en esa posesión anterior.
  - Excepción: no se absorben las que siguen a un balón muerto, porque tras una reanudación «hay un poseedor claro» (`CHALLENGE.md`).
- (c) Ajustar los costes del Viterbi **conjuntamente** sobre los escenarios v2 y pesimista, para que el criterio de selección mida una sola configuración entregable.
  - Ahora `it3c.json` lleva los costes ajustados en el pesimista (`a_muerto 32`).
  - Los de v2 eran `a_muerto 1, sesgo_d 2`.

**Disponibilidad y licencia de Metrica sample-data.**
- Repositorio público: `github.com/metrica-sports/sample-data`.
- No tiene fichero LICENSE. El README («Legal stuff») pide un uso responsable y reconocer la fuente si se usa en algo público.
- Cumple RULES.md: es un «open dataset» accesible para los organizadores.
- Hay que citarlo en el documento técnico.

Ficheros (tamaños en bytes):

```
data/Sample_Game_1/Sample_Game_1_RawEventsData.csv            139446
data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv 32782961
data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv 32782073
data/Sample_Game_2/Sample_Game_2_RawEventsData.csv            153897
data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv 31890338
data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv 29627711
data/Sample_Game_3/Sample_Game_3_events.json                  1709486
data/Sample_Game_3/Sample_Game_3_metadata.xml                    81806
data/Sample_Game_3/Sample_Game_3_tracking.txt                 53443115
documentation/events-definitions.pdf
```

URL base: `https://raw.githubusercontent.com/metrica-sports/sample-data/master/`.

El tracking de SG3 está en formato EPTS FIFA, el mismo `frame:jugadores:balón` que el ATD del reto. Ninguno de los tres partidos (SG1, SG2, SG3) trae estado del balón ni posesión en el tracking: las etiquetas salen de los eventos.

### P2. Degradación realista de los huecos de balón (día 2)

**Por qué.**
- Según la segunda tabla del diagnóstico, en el ATD los huecos empiezan con el balón a 1,45 m del jugador más cercano; en nuestra degradación, a 2,3–2,6 m.
- En el ATD los huecos empiezan dentro de un área el doble de veces que un frame cualquiera.
- El ATD no tiene huecos de 1–5 s.
- Con huecos al azar, el modelo aprende que el inicio de un hueco no dice nada. En el ATD sí dice algo: «recepción o duelo», que es donde se decide la posesión.
- Lo que más mueve la validación (`docs/HIBRIDO.md`, punto 2) es precisamente dónde y cuándo falta el balón.
- Se puede calibrar solo con agregados del input.

### P3. Pérdida ponderada en los bordes (día 3, si queda tiempo)

**Por qué.** El spellF1 se decide en los bordes. El 18 % de los frames de IDSSE está a menos de 1 s de una transición, pero la entropía cruzada los pesa igual que el interior. Ponderar ×3 los pasos a ±1 s de una transición del GT reetiquetado son unas 10 líneas en `entrenar.py` y un run de 13 min.

### Descartado para it4

- **Convenciones de D** (cuándo empieza y acaba el balón muerto). Los tramos D de DFL duran ~23 s de media y los del GT público, ~15 s (n = 4). No hay una regla defendible para reetiquetarlos sin mirar el GT del reto. Se pregunta a los organizadores (sección 7) y se cubre con el ajuste conjunto v2 + pesimista.
- **Smart Tagging en el decodificador.** No existe ni en IDSSE ni en Metrica sample; habría que simularlo. Queda para después de it4.
- **PFF / SkillCorner.** PFF necesita un formulario; SkillCorner va a 10 fps. Ahora no.

## 3. Pasos concretos

Nombres:
- Candidatos: `it4a` = P1, `it4b` = P2, `it4c` = P3.
- Runs en `salidas/hibrido/runs/<run>/`.
- JSON de CV en `salidas/hibrido/<run>_<sufijo>.json`.
- Configs entregables en `modelos/hibrido/<run>.json` y `.pt`.

### Paso 1. Preparación (20 min)

1. Crea `conftest.py` en la raíz del repo:
   ```python
   import os, sys
   sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
   ```
   Comprueba que `.venv/bin/pytest -q` da 10 passed.
2. Ejecuta `ssh marenostrum 'squeue -u $USER'` y anota si hay algo en cola. Si `tts-esti` está en `acc_debug`, los `sbatch` de abajo quedarán esperando en cola; no pasa nada.
3. No hace falta añadir nada a `.gitignore`: `data/*` y `salidas/` ya están. Los datos de Metrica van a `$MTAH_EXTERNAL_DIR/metrica/`, fuera del repo.

### Paso 2. Fuente Metrica sample (2 h)

**`scripts/descargar_metrica.py`** (copia el patrón de `scripts/descargar_idsse.py`):
- Baja los 9 ficheros de arriba a `$MTAH_EXTERNAL_DIR/metrica/Sample_Game_{1,2,3}/`.
- Reanuda descargas a medias y comprueba el tamaño contra la lista de arriba.
- El docstring lleva la cita (Metrica Sports, sample-data, GitHub) y la nota del README sobre reconocer la fuente.

**`mtah/fuentes/metrica.py`**:
- `PARTIDOS = ("SG1", "SG2", "SG3")`.
- `cargar(id, usar_cache=True)` devuelve un `Partido`, cacheado en `$MTAH_EXTERNAL_DIR/cache/metrica_<id>.npz`.
- `disponibles()`.
- Sigue el patrón de `mtah/fuentes/idsse.py`: dos partes, `SEPARACION = 25` frames X entre partes, `seccion` 1/2 y `campo = (105.0, 68.0)`.

Tracking:
- **SG1/SG2:**
  - Carga con `kloppy.metrica.load_tracking_csv(home_data=..., away_data=..., coordinates="metrica")` y convierte con `dataset.to_df()`. Columnas: `frame_id, period_id, ball_x, ball_y, <player_id>_x, <player_id>_y`.
  - El equipo de cada jugador sale de `dataset.metadata.teams` (el primero es el local) → `.players[*].player_id`.
  - Si kloppy da problemas, parsea con pandas: `pd.read_csv(ruta, skiprows=2)`. Las columnas son `Period, Frame, Time [s]`, luego pares x,y por jugador y `Ball` al final, con NaN donde no hay observación.
- **SG3:**
  - Carga con `kloppy.metrica.load_tracking_epts(meta_data=xml, raw_data=txt, coordinates="metrica")`.
  - Tiene 11 bloques `DataFormatSpecification` porque el orden de columnas cambia con cada sustitución: no lo parsees a mano.
  - Periodos: la 1ª parte va de los frames 1 a 69661 y la 2ª, de 69662 a 143761.
- Comprueba que `coordinates="metrica"` deja el rango 0–1 con la y hacia abajo: la media de `ball_x` debe salir ≈ 0,5 y la x de los porteros en los extremos. No hace falta conversión.

Eventos → GT por frame:
- Función pura `etiquetas_desde_eventos(eventos, n_frames) -> int8[n]` en el mismo módulo.
- Recibe `eventos = [(equipo 0/1, tipo, subtipos_str, inicio, fin)]`, ya normalizados y ordenados por `(inicio, índice)`.

Formato de los eventos:
- **SG1/SG2:** CSV con las columnas `Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,Start X,Start Y,End X,End Y`.
  - `Team` ∈ {Home, Away}.
  - Ejemplo de fila: `Away,BALL OUT,,1,1374,54.96,1425,57,Player15,,0.87,0.29,1.05,0.17`.
  - El primer KICK OFF trae `End Frame 0`: usa el inicio.
- **SG3:** JSON `{"data": [...]}`. Cada evento tiene `team.id` (FIFATMA = local, FIFATMB = visitante), `type.name`, `subtypes` (null, un objeto o una lista de `{name,id}`), `period`, `start.frame` y `end.frame`.
- Tipos: `SET PIECE, PASS, CARRY, SHOT, RECOVERY, BALL LOST, BALL OUT, CHALLENGE, FAULT RECEIVED, CARD`.

Reglas (cada parte empieza en D):

| evento | efecto |
|---|---|
| `SET PIECE` | Vivo desde `inicio`, con el equipo del evento. |
| `PASS`, `CARRY`, `SHOT`, `RECOVERY` | Equipo del evento desde `inicio`. Si el estado era D, pasa a vivo igualmente (set piece sin anotar); cuéntalo en un contador e imprímelo. |
| `BALL LOST` | Sin cambio: la posesión sigue con quien la pierde hasta el primer evento del rival («contested or loose, possession stays with the last team», `CHALLENGE.md`). Si el subtipo contiene `END HALF`, D desde `inicio`. |
| `BALL OUT` | D desde `fin`: el evento va del último toque al cruce de la línea. Si `fin <= inicio`, usa `inicio`. |
| `SHOT` con `GOAL` en los subtipos (y sin `WOODWORK`) | D desde `fin`. |
| `FAULT RECEIVED` | D desde `inicio`. |
| `CHALLENGE`, `CARD` | Se ignoran. |

Tras el último evento de la parte se mantiene el estado hasta el final de la parte. El resultado va al `gt` del `Partido`; valídalo con `p.validar()`.

**`tests/test_metrica.py`**: a partir de una lista sintética de eventos, comprueba que:
- los frames previos al kick-off son D;
- el set piece fija el equipo;
- `BALL LOST` no cambia nada;
- `RECOVERY` del rival cambia la posesión en su frame;
- `BALL OUT` pone D desde `fin`;
- `FAULT RECEIVED` pone D;
- `SHOT ... GOAL` pone D hasta el siguiente `SET PIECE`.

**`scripts/estadisticas_convencion.py`** (solo agregados; no toca el reto):
- Fuentes: IDSSE original, IDSSE reetiquetado con T ∈ {1, 2, 3} s y Metrica SG1–3.
- Imprime, por partido y en media: fracción D, tramos D por hora y su duración media, posesiones por minuto de balón vivo, duración media de posesión y fracción de posesiones de menos de 3 s.

Debe reproducir estos valores de referencia de IDSSE, ya medidos (±0,1):

| | pos./parte | duración media | pos./min vivo | frames cambiados |
|---|---|---|---|---|
| original | 218 | 7,2 s | ~9 | — |
| T = 1 s | 183 | 8,8 s | 7,1 | 0,5 % |
| T = 2 s | 144 | 11,2 s | 5,6 | 1,7 % |
| T = 3 s | 119 | 13,7 s | 4,6 | 3,1 % |

Comprobaciones extra de Metrica que hay que imprimir, por partido:
- la fracción de frames con el balón en NaN;
- la fracción de frames con el balón congelado (`mtah.degradar.tramos_congelados`) y si coincide con D.

`degradar` ya quita el congelado. Si Metrica tiene el balón en NaN durante D, anótalo en `docs/HIBRIDO.md`: es un indicio sobre la convención de Metrica, y mirarlo es legítimo porque es otro partido.

**Elección de T (prerregistrada, sin mirar BAR-ATH).**
- T es el valor de {1, 2, 3} s con el que las posesiones por minuto de balón vivo de IDSSE reetiquetado quedan más cerca de la media de Metrica SG1–3.
- Si Metrica no se puede descargar o parsear en el tiempo previsto (tope: 2,5 h desde que empieza el paso 2), se usa T = 2 s a priori («a player keeps the ball for a few seconds», `CHALLENGE.md`) y se sigue sin el dominio Metrica.
- No se prueban más valores de T en ninguna fase.

### Paso 3. Reetiquetado, escenarios y flags (1 h)

**`mtah/convencion.py`**:

```python
MINIMO_S_DEFECTO = 2.0
def posesion_controlada(gt, seccion, minimo_frames):
    """Absorbe en la posesión anterior las posesiones H/A cortas que le siguen.

    Regla: una posesión H/A de menos de minimo_frames, precedida (sin D ni X entre medias) por una
    posesión viva del otro equipo, se absorbe en esa posesión anterior.
    Recorrido: una sola pasada de izquierda a derecha por sección, guardando el último segmento
    fusionado. Si el segmento actual es H/A, corto, y el anterior es vivo y del otro equipo, se
    extiende el anterior. Si es del mismo equipo y contiguo, se fusionan."""
```

Usa solo numpy y devuelve una copia. Trabaja por sección (`seccion == 1` y `== 2`) y respeta X: X cuenta como un segmento más y rompe la cadena.

**`tests/test_convencion.py`**, con T = 50:
- `A(100) B(30) A(100)` → todo A.
- `A(100) D(50) B(30) A(100)` → B se queda, porque viene tras D.
- `A(100) B(60) A(100)` → sin cambio.
- `A(100) B(30) B(30)` → fusiones de segmentos contiguos.
- `A(100) B(20) A(20) B(20) A(100)` → todo A.
- `A(100) B(30) D(50)` → B se absorbe en A (toque sin control antes de salir).

**`mtah/escenarios.py`**: un sitio único para los escenarios de evaluación, que ahora están duplicados en `entrenar.py`, `afinar.py` y `evaluar_externo.py`:

```python
PESIMISTA = dict(congelado_ausente=0.0, congelado_fuera_ausente=0.0)
ESCENARIOS = {"v2": {}, "pes": PESIMISTA}      # P2 añade "v3" y "v3pes"
SUFIJO = {"v2": "", "pes": "_pes"}             # nombres de fichero compatibles con los runs it3*
```

**`scripts/hibrido/entrenar.py`**:
- Añade `--convencion {dfl,metrica}` (por defecto `dfl`) y `--minimo_s` (por defecto `convencion.MINIMO_S_DEFECTO`). Con `metrica`, en `_preparar` haz `p = replace(p, gt=posesion_controlada(p.gt, p.seccion, int(round(minimo_s * p.fps))))` **antes** de `R.rasgos(...)`.
- Añade `--escenarios_eval v2,pes`, que sustituye a `--pesimista`. Mantén `--pesimista` como alias de `v2,pes`. Los OOF se guardan como `oof_<m>_s<s><SUFIJO>.npz`.
- Guarda el modelo de cada fold como `fold_<m>.pt` (2,4 MB cada uno), con el mismo formato que `final.pt`. El paso 8 lo necesita para evaluar un run antiguo en escenarios nuevos sin reentrenar.
- Añade el modo `--solo_eval <run_previo>`: carga los `fold_<m>.pt` de ese run y escribe los OOF de los `--escenarios_eval` pedidos, sin entrenar. Corre en la CPU local en minutos.
- Declara ya `--peso_borde` y `--radio_borde` (P3, paso 9), con valores neutros: 1.0 y 5.

**`scripts/hibrido/afinar.py`**:
- Añade `--convencion` y `--minimo_s`, con el mismo significado. Se aplican al `gt` leído de la caché de rasgos en `items_base`; la caché guarda el GT original, así que no hace falta regenerarla.
- Añade `--escenarios v2,pes`:
  - `ITEMS` incluye los ítems de todos los escenarios listados, con clave `(partido, semilla, escenario)`.
  - El objetivo del ascenso es la media de spellF1 sobre todos ellos, así que sale **un solo juego de costes por fold**.
  - El JSON de salida añade `spell_f1_por_escenario` y `macro_f1_por_escenario` (media de las filas de cada escenario), además de la media global.
- Añade 48 y 64 a `REJILLA["cambio"]`: con el GT reetiquetado, el coste de cambio óptimo puede subir.
- Con un solo escenario, mantén el comportamiento actual (compatibilidad con `salidas/hibrido/it3*.json`).

**`scripts/evaluar_externo.py`**:
- Añade `--fuente idsse|metrica`, `--escenarios` y `--convencion/--minimo_s` (estos dos solo afectan al GT de IDSSE).
- Sirve para el zero-shot en Metrica con `hibrido_<run>`. El predictor corre la red en CPU: ~1–2 min por partido y escenario.

Tests: amplía `tests/test_hibrido.py` con un test que compruebe que `afinar` con `--escenarios v2,pes` devuelve un único `Costes` por fold. Se puede probar `objetivo` con ítems sintéticos.

### Paso 4. Run it4a en MareNostrum (se lanza y se sigue en background)

```bash
# desde el worktree, con los tests en verde
rsync -a --delete --exclude __pycache__ mtah scripts marenostrum:mtah2027/code/
ssh marenostrum 'cd $HOME/mtah2027 && sbatch --output=$HOME/mtah2027/logs/%x-%j.out \
  --export=ALL,RUNS="it4a:--mezcla --escenarios_eval v2,pes --convencion metrica --minimo_s <T> --final" \
  code/scripts/hibrido/entrenar_mn5.sbatch'
```

`<T>` es el valor fijado en el paso 2. Se conservan los argumentos de it3c, que son los valores por defecto: `--semillas 6 --epocas 30 --lotes_epoca 64 --lote 32 --largo 512 --lr 1e-3 --d 64 --dt 128`.

Seguimiento sin `sleep`:
- Lanza un proceso en background con `until ! ssh marenostrum 'squeue -h -u $USER -n mtah-hibrido' | grep -q . ; ...`, o usa el monitor del agente.
- Mira el log con `ssh marenostrum 'tail -3 $HOME/mtah2027/logs/mtah-hibrido-<jobid>.out'`.
- Duración esperada: unos 13–15 min desde que arranca.

Al acabar:

```bash
rsync -a marenostrum:mtah2027/runs/it4a salidas/hibrido/runs/
cp salidas/hibrido/runs/it4a/final.pt modelos/hibrido/it4a.pt
```

Comprueba que hay 28 ficheros `oof_*` (7 partidos × 2 semillas × 2 escenarios) y `resumen.json`.

### Paso 5. Referencia it3c con el protocolo nuevo (local, en background, mientras corre el paso 4)

Usa los OOF que ya hay en `salidas/hibrido/runs/it3c/` (todos tienen su versión `_pes`):

```bash
.venv/bin/python scripts/hibrido/afinar.py salidas/hibrido/runs/it3c --escenarios v2,pes \
   --convencion metrica --minimo_s <T> --salida salidas/hibrido/it3c_M.json
.venv/bin/python scripts/hibrido/afinar.py salidas/hibrido/runs/it3c --escenarios v2,pes \
   --convencion dfl --salida salidas/hibrido/it3c_conj.json
```

- `it3c_M` es la referencia del criterio.
- `it3c_conj` va a la tabla, para ver cuánto cuesta el ajuste conjunto frente a los 0,717/0,616 por separado.
- Guarda `modelos/hibrido/it3c_m.json`, con los pesos `it3c.pt` y los `costes_final` de `it3c_M.json`. Es el fallback si it4a no pasa.

### Paso 6. Evaluación de it4a (30 min)

```bash
.venv/bin/python scripts/hibrido/afinar.py salidas/hibrido/runs/it4a --escenarios v2,pes \
   --convencion metrica --minimo_s <T> --salida salidas/hibrido/it4a_M.json
```

Escribe `modelos/hibrido/it4a.json` con `pesos_red: it4a.pt` y los `costes_final` de `it4a_M.json`. Hace falta para el zero-shot aunque luego se rechace.

Zero-shot en Metrica (si se consiguió en el paso 2), para `hibrido_it3c_m` y `hibrido_it4a`:

```bash
.venv/bin/python scripts/evaluar_externo.py hibrido_it3c_m hibrido_it4a --fuente metrica --escenarios v2,pes --semillas 2
```

Anota el spellF1 y el macroF1 medios (3 partidos × 2 semillas × 2 escenarios) y la proporción pred/gt.

### Paso 7. Decisión y documentación de P1 (30 min)

- Aplica la regla de la sección 4 **tal cual**.
- Añade la tabla a `docs/HIBRIDO.md` (sección 6).
- Escribe `docs/PREGUNTA-ORGANIZADORES.md` con el borrador de la sección 7.
- Comprueba que `.venv/bin/pytest -q` está en verde.

Fin del día 1. **Todavía no ejecutes `generar_entrega.py`.**

### Paso 8. P2, huecos realistas (día 2: 3–4 h más 15 min de MN5)

Añade campos nuevos a `Degradacion` en `mtah/degradar.py`, todos con un valor neutro por defecto para que se reproduzca el comportamiento anterior. La caché de rasgos cambia de nombre porque `version()` calcula el crc32 de `repr(Degradacion)`; se regenera sola en ~2 min.

```python
hueco_peso_cerca: float = 0.0    # peso extra si el jugador más cercano está a < 2 m del balón
hueco_peso_area: float = 0.0     # peso extra si el balón está en un área (x < 0.17 o x > 0.83, 0.2 < y < 0.8)
hueco_peso_fuera: float = 0.0    # peso extra si el balón está fuera del campo
hueco_minimo_largo: int = 1      # longitud mínima de los huecos "largos" (el ATD casi no tiene de 1-5 s)
hueco_corto_prob: float = 0.0    # probabilidad de que un hueco sea corto (2-24 frames) en vez de largo
```

Cambios en `_quitar_balon`:
- **Inicio del hueco:** muestréalo con `rng.choice(candidatos, p=w/w.sum())`.
  - Pesos: `w = 1 + peso_cerca·[dmin<2] + peso_area·[área] + peso_fuera·[fuera]`.
  - Se calculan sobre el balón ya sin congelado.
  - Los frames NaN tienen peso 0, salvo que no quede otro.
- **Longitud:** con probabilidad `hueco_corto_prob`, uniforme entre 2 y 24 frames. Si no, lognormal como ahora, con `max(largo, hueco_minimo_largo)`.

`scripts/hibrido/calibrar_huecos.py` (copia el patrón de `calibrar_balon.py`):
- Vector de agregados:
  - fracción de huecos que empiezan en un área;
  - fracción que empiezan fuera del campo;
  - mediana de dmin (m)/5;
  - media de jugadores a menos de 5 m, /5;
  - fracción de huecos de menos de 1 s;
  - fracción de huecos de 1–5 s;
  - fracción de frames sin balón que caen en huecos de más de 20 s;
  - más los 10 de `calibrar_balon.agregados`.
- Objetivo del ATD medido el 7-nov: `[0.30, 0.15, 1.45/5, 1.60/5, 0.11, 0.007, 0.67]`. El script debe recalcularlo desde `reto.cargar(gt_publico=False)` (es input, no etiquetas) e imprimir solo esos números.
- Rejilla, sobre 3 partidos IDSSE; gana el L1 mínimo:
  - `peso_cerca ∈ {0,2,4,8}`
  - `peso_area ∈ {0,1,2,4}`
  - `peso_fuera ∈ {0,2,4}`
  - `congelado_fuera_ausente ∈ {0.5,0.7,0.9}`
  - fijos: `hueco_corto_prob = 0.1`, `hueco_minimo_largo = 125`
- El ganador se fija como escenario `v3` en `mtah/escenarios.py`, con `v3pes = v3 + PESIMISTA`.
- Test en `tests/test_base.py`: con `peso_area=8`, la fracción de huecos que empiezan en un área sube frente a `0`; además, determinismo.

`entrenar.py`: añade `--deg_train v2|v3` (por defecto `v2`). El dict del escenario se combina con los `NIVELES` de `--mezcla`: `{**ESCENARIOS[deg_train], **nivel}`.

Runs:

```bash
# it4a evaluado en los escenarios nuevos, sin reentrenar (local, CPU)
.venv/bin/python scripts/hibrido/entrenar.py --run it4a --salida salidas/hibrido/runs --solo_eval it4a \
   --escenarios_eval v3,v3pes --convencion metrica --minimo_s <T> --procesos 8
# it4b en MN5
rsync -a --delete --exclude __pycache__ mtah scripts marenostrum:mtah2027/code/
ssh marenostrum '... --export=ALL,RUNS="it4b:--mezcla --deg_train v3 --escenarios_eval v2,pes,v3,v3pes --convencion metrica --minimo_s <T> --final" ...'
```

Después:
- Ejecuta `afinar.py` para it4a e it4b con `--escenarios v2,pes,v3,v3pes --convencion metrica`, con salidas `it4a_M4.json` e `it4b_M4.json`.
- Haz el zero-shot en Metrica de `hibrido_it4b` con `--escenarios v2,pes,v3,v3pes`.
- Decide con la regla de la sección 4, usando el criterio de 4 escenarios.

### Paso 9. P3, pérdida en los bordes (día 3: 1 h más 15 min de MN5)

`entrenar.py`: añade `--peso_borde W` (por defecto 1.0) y `--radio_borde R` en pasos (por defecto 5, que es 1 s a 5 fps). En el bucle:
- Usa `ce = CrossEntropyLoss(ignore_index=-1, reduction="none")`.
- Peso por paso: `w = 1 + (W-1)·[hay un cambio de etiq en ±R pasos dentro de la ventana]`, calculado con `etq` e ignorando los −1.
- Pérdida: `perdida = (ce·w).sum() / w[etq>=0].sum()`.

El run `it4c` es el mejor de {it4a, it4b} con `--peso_borde 3`. Se evalúa igual y con la misma regla que su base.

### Paso 10. Cierre de it4 (día 3)

1. Elige el modelo final con la regla de la sección 4.
2. Si es un `it4*`, deja `modelos/hibrido/it4<x>.json` y su `.pt`. Saca de `modelos/` los `it4*.json` de los candidatos rechazados (si no, el registro de predictores los expondrá) y anótalo.
3. Haz la comprobación de alineación, **una sola vez**, con el modelo elegido:
   ```bash
   .venv/bin/python scripts/generar_entrega.py hibrido_<elegido>
   ```
   - Anota el spellF1 y el macroF1 públicos, el número de posesiones y el sha256.
   - Si el spellF1 público cae por debajo de 0,33 (lo que saca smooth50 en esa ventana) o falla la validación, es un bug de alineación. Se arregla el bug, no el modelo, y se vuelve a validar el fichero; eso no es ajustar.
4. Comprueba que `.venv/bin/pytest -q` está en verde y que `git status` no muestra nada de `data/` ni de `salidas/`.

## 4. Criterio de selección (fijado ahora; no se toca)

Definiciones:
- **GT-M:** el GT de IDSSE reetiquetado con `posesion_controlada` y la T del paso 2. Todas las cifras del criterio se calculan con GT-M y con el scorer oficial, vía `afinar.py`:
  - CV anidada por partido;
  - un solo juego de costes por fold, ajustado conjuntamente sobre los escenarios del criterio.
- **S(modelo):** media del spellF1 de CV anidada sobre los escenarios del criterio. Son `{v2, pes}` para P1 y `{v2, pes, v3, v3pes}` para P2 y P3, con los dos candidatos evaluados sobre los mismos ítems.
- **Z(modelo):** spellF1 zero-shot medio en Metrica SG1–3 (2 semillas × escenarios del criterio), con los costes del ajuste con los 7 partidos de IDSSE. Si Metrica no está disponible, Z no se usa.

En cada paso se compara el candidato `C` con el mejor hasta ese momento, `B` (al principio, `B = it3c_m`):
1. Se acepta `C` si `S(C) − S(B) ≥ +0,02` **y**, si hay Z, `Z(C) ≥ Z(B) − 0,02`.
2. Si `|S(C) − S(B)| < 0,02`, se acepta `C` solo si `Z(C) − Z(B) ≥ +0,03`. Si no, se queda `B`.
3. Si `S(C) < S(B) − 0,02`, se rechaza.
4. El macroF1 se reporta y solo desempata si las reglas anteriores dejan un empate exacto.

Regla de parada:
- Si it4a se rechaza **y** `Z(it4a) ≤ Z(it3c_m)`, la hipótesis de la convención no se sostiene y se abandona P1.
  - Se entrega `it3c_m`. Excepción: si `S(it3c_m)` con la convención de DFL queda más de 0,02 por debajo de los 0,717/0,616 por separado, se entrega `it3c` tal cual.
  - P2 y P3 se ejecutan con `--convencion dfl`.
- Si P2 y P3 tampoco pasan, se entrega el mejor `B`, y it4 termina igualmente con la tabla y la comprobación final.

Por qué 0,02: en `docs/HIBRIDO.md`, las mejoras aceptadas entre iteraciones fueron de ese orden, sobre 7 partidos × 2 semillas emparejados. El ruido de 0,04 de `EVALUATION.md` es para un solo partido.

## 5. Tiempo y orden

| cuándo | paso | tiempo | sale |
|---|---|---|---|
| día 1 (7-nov) | 1. Preparación | 0:20 | tests verdes con `pytest -q` |
| | 2. Fuente Metrica, estadísticas y T | 2:00–2:30 | `mtah/fuentes/metrica.py`, T fijada |
| | 3. Reetiquetado, escenarios y flags | 1:00 | `mtah/convencion.py`, `mtah/escenarios.py` |
| | 4. Lanzar it4a en MN5 | 0:10, más 15 min de cola y run en background | `runs/it4a` |
| | 5. Referencia it3c (local, en paralelo) | 0:20 de CPU | `it3c_M.json`, `it3c_conj.json` |
| | 6. Evaluación de it4a y zero-shot | 0:30 | `it4a_M.json`, Z |
| | 7. Decisión, docs y borrador de la pregunta | 0:30 | tabla en HIBRIDO.md |
| día 2 (8-nov) | 8. P2, huecos realistas | 3:00–4:00 más MN5 | `v3`, `it4b` |
| día 3 (9-nov) | 9. P3, pérdida en los bordes | 1:00 más MN5 | `it4c` |
| | 10. Cierre, comprobación única y entrega | 0:40 | `modelos/hibrido/it4*.json`, sha256 |

Total del día 1: ~6 h. Si el paso 2 se pasa del tope de 2,5 h, aplica el fallback T = 2 s y sigue: lo importante del día 1 es tener `it4a` evaluado.

## 6. Entregables

1. `docs/HIBRIDO.md`: una sección nueva «it4» con:
   - (a) la tabla de convenciones del paso 2 (IDSSE original y con cada T, Metrica SG1–3);
   - (b) la T elegida y por qué;
   - (c) la tabla de resultados:

     | modelo | GT | costes | v2 | pes | v3 | v3pes | S | Z (Metrica) | macroF1 | pred/gt |
     |---|---|---|---|---|---|---|---|---|---|---|
     | it3c (HIBRIDO.md) | DFL | separados | 0,717 | 0,616 | — | — | 0,667 | — | 0,94/0,90 | |
     | it3c_conj | DFL | conjuntos | | | | | | | | |
     | it3c_m | GT-M | conjuntos | | | | | | | | |
     | it4a | GT-M | conjuntos | | | | | | | | |
     | it4b … | | | | | | | | | | |

   - la decisión tomada según la sección 4;
   - la comprobación de alineación final, en una línea;
   - los agregados de la predicción en BAR-ATH (número de posesiones, duración media, fracción D, posesiones de menos de 3 s), comparados con los de it3c (357 posesiones; 27–30 % de menos de 3 s).
2. `modelos/hibrido/it4<x>.json` e `it4<x>.pt` si mejora. Si no, `it3c_m.json` o `it3c.json` según la regla de parada.
3. `salidas/hibrido_<elegido>.csv` validado (lo ignora git), con su sha256 en HIBRIDO.md.
4. `docs/PREGUNTA-ORGANIZADORES.md` (borrador, sin enviar).
5. `.venv/bin/pytest -q` en verde, con los tests nuevos (`test_convencion.py`, `test_metrica.py` y los huecos en `test_base.py`).
6. Citas de IDSSE (ya está) y de Metrica sample-data en `docs/HIBRIDO.md`.

## 7. Borrador de pregunta a los organizadores (no enviar)

Guárdalo en `docs/PREGUNTA-ORGANIZADORES.md`. El equipo decide si se manda por correo (`hackathon.talentarena@mobileworldcapital.com`) o como issue.

> **Subject: Dead-ball boundaries and "controlled possession" in the corrected tracking**
>
> We train and validate only on other matches (IDSSE/DFL, Metrica sample data). We have noticed that the labelling conventions differ between providers in two places that matter for spellF1. We are not asking about BAR-ATH labels. We are asking only about the general convention of the corrected tracking that the ground truth comes from:
>
> 1. **Dead ball start/end.** Is `D` marked from the moment the ball crosses the line or the whistle blows until the restart touch? Or does it include a margin, for example while the ball is carried to the restart spot? In DFL data, dead-ball stretches average about 23 s. In the public window they average about 15 s.
> 2. **Short touches.** Sometimes a defender deflects or clears a pass without control, and the original team gets the ball back within 1–2 s. Does the corrected tracking keep the original team as owner the whole time, as the "contested or loose" rule in CHALLENGE.md suggests? Or does possession switch and switch back?
> 3. **Restarts lost immediately.** After a throw-in or free kick that is lost within a second, does the restarting team get a possession of its own?
>
> Thanks. A one-line answer to each question is enough for us to align our external labels.

## 8. Notas para el ejecutor

- Nunca imprimas filas de `data/tactical/*` ni de `data/cache/BAR-ATH_tactical.npz`. Usa `reto.cargar()` solo dentro de scripts que impriman agregados (`calibrar_*`, `estadisticas_convencion`, `generar_entrega`).
- En `salidas/hibrido/runs/it3c/`, los ficheros `oof_*` del escenario pesimista llevan el sufijo `_pes`. Respeta `SUFIJO` para que `afinar.py` los encuentre.
- `mtah/predictores/__init__.py` registra como predictor cualquier `modelos/hibrido/*.json`. Antes de la entrega, saca de esa carpeta (o mueve a `salidas/`) los candidatos rechazados.
- MareNostrum:
  - El entorno es `singularity`, con `torch 2.11+cu128` y `numpy 2.5.4`.
  - El código se copia con `rsync -a --delete --exclude __pycache__ mtah scripts`; nada de `data/`.
  - La caché de IDSSE ya está allí, y el reetiquetado se hace al cargar, así que no hay que subir nada nuevo.
  - Si más adelante usas Metrica para entrenar, sube solo `external/cache/metrica_*.npz`.
- Si `squeue` muestra `tts-esti` en `acc_debug`, nuestro trabajo quedará en `PENDING`: sigue con los pasos locales y vuelve a mirar más tarde.
