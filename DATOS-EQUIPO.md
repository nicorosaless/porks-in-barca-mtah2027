# Cómo conseguir los datos del partido

Guía interna del equipo para tener los datos del Barça–Athletic en tu máquina. Los detalles
oficiales están en [DATA.md](DATA.md). Si esta guía y DATA.md no coinciden, manda DATA.md.

## Antes de empezar

- **Cada uno se descarga los datos con su propio enlace.** Los datos son solo para participantes
  inscritos que han aceptado los términos de uso. No nos pasamos los ficheros ni el enlace entre
  nosotros: el enlace es personal.
- **Espacio en disco:** ~5,7 GB para `tactical/`. Si también bajas el vídeo broadcast, ~14 GB más.

## 1. Inscríbete y recibe el enlace

1. Inscríbete en
   [More Than A Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027) con el
   mismo nombre de equipo que el resto.
2. Lee y firma los términos y condiciones del formulario.
3. Recibirás por correo un enlace de descarga (una carpeta de Google Drive).

## 2. Descarga los ficheros

Clona el repo del equipo y descarga dentro de `data/`:

```bash
git clone https://github.com/nicorosaless/more-than-a-hack-2027.git
cd more-than-a-hack-2027
```

### Qué bajar

| carpeta | fichero | tamaño | ¿hace falta? |
|---|---|---|---|
| `tactical/` | `BAR-ATH_tactical_atd.txt` | 195,7 MB | **sí** |
| `tactical/` | `BAR-ATH_tactical_atd.xml` | 59 KB | **sí** |
| `tactical/` | `BAR-ATH_tactical_smart_tagging.xml` | 261 KB | **sí** |
| `tactical/` | `BAR-ATH_tactical_homography.csv` | 23,2 MB | **sí** |
| `tactical/` | `BAR-ATH_tactical.mp4` | 5,2 GB | si vas a trabajar con vídeo |
| `broadcast/` | `BAR-ATH_broadcast_atd.txt` | 142,0 MB | opcional |
| `broadcast/` | `BAR-ATH_broadcast_atd.xml` | 65 KB | opcional |
| `broadcast/` | `BAR-ATH_broadcast_homography.csv` | 15,4 MB | opcional |
| `broadcast/` | `BAR-ATH_broadcast.mp4` | 14,0 GB | **no**, de momento |

El plan actual se apoya en el ATD táctico y el Smart Tagging. El broadcast no está sincronizado con
el vídeo táctico y por ahora no lo usamos. No bajes sus 14 GB salvo que tu línea de trabajo lo
necesite.

### Cómo bajarlos

**Navegador (lo más simple).** Abre el enlace de Drive y descarga cada fichero por separado. Si
descargas la carpeta entera, Drive la parte en varios `.zip` y es más fácil que falle.

**Ficheros grandes (los `.mp4`).** Con el navegador, si la descarga se corta hay que empezar de
cero. Si te pasa, usa `rclone` con tu cuenta de Google. Ojo: rclone tampoco reanuda un fichero a
medias, pero reintenta solo y verifica el tamaño:

```bash
rclone config                    # crea un remoto "gdrive" de tipo drive con tu cuenta
# Añade la carpeta compartida a tu Drive ("Añadir acceso directo a Drive") y luego:
rclone copy "gdrive:<carpeta>/BAR-ATH_tactical.mp4" data/tactical/ --progress --retries 10
```

No uses servicios de terceros que descarguen en tu nombre (Composio, conversores online,
etc.): el fichero pasaría por sus servidores.

### Dónde dejarlos

Respeta exactamente los nombres y la estructura:

```
data/
├─ tactical/
│  ├─ BAR-ATH_tactical.mp4
│  ├─ BAR-ATH_tactical_atd.xml
│  ├─ BAR-ATH_tactical_atd.txt
│  ├─ BAR-ATH_tactical_homography.csv
│  └─ BAR-ATH_tactical_smart_tagging.xml
└─ broadcast/
   ├─ BAR-ATH_broadcast_atd.xml
   ├─ BAR-ATH_broadcast_atd.txt
   └─ BAR-ATH_broadcast_homography.csv
```

¿Prefieres guardarlos en otro disco? Apunta `MTAH_DATA_DIR` a esa carpeta y no hay que tocar nada
más:

```bash
export MTAH_DATA_DIR=/ruta/a/mtah2027
```

## 3. Comprueba que están bien

Desde la raíz del repo (o desde `$MTAH_DATA_DIR`), ejecuta lo siguiente. Los ficheros que no hayas
bajado salen como `FAILED open or read`; es normal.

```bash
cd data && md5sum -c - <<'EOF'
fa362d5aaea406388fbe6ab762d2ecc7  tactical/BAR-ATH_tactical_atd.txt
5d4ca14ac41307e7692fb7c50a211175  tactical/BAR-ATH_tactical_atd.xml
5f8b7e6e0b3b50e3b325c35986340cac  tactical/BAR-ATH_tactical_homography.csv
b424cde000cbc1bfe558c12458340c3a  tactical/BAR-ATH_tactical.mp4
987cfb2a1b2d120f05daf2f7f902ead8  tactical/BAR-ATH_tactical_smart_tagging.xml
f8579af8f45a264e149c94a556cbf49c  broadcast/BAR-ATH_broadcast_atd.txt
a38c2fda72303727e57054267960397b  broadcast/BAR-ATH_broadcast_atd.xml
74687c8650200959ffb55ba5c3faa479  broadcast/BAR-ATH_broadcast_homography.csv
EOF
cd ..
```

En macOS, `md5sum` no viene de serie: `brew install coreutils` o usa `md5 -r <fichero>` y compara a
mano. Para el `BAR-ATH_broadcast.mp4` no hay checksum aquí; comprueba que pesa exactamente
13 962 360 320 bytes.

Después, prueba que el código encuentra los datos:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-starter.txt
python starter/load_smart_tagging.py          # lee el Smart Tagging y comprueba los kick-offs
python evaluation/run_evaluation.py           # puntúa los baselines contra el GT público
```

## 4. Entorno del equipo y datos externos (IDSSE)

El código del equipo vive en `mtah/` y `scripts/`. El plan está en
[docs/PLAN.md](docs/PLAN.md) y el estado del arte en [docs/RESEARCH.md](docs/RESEARCH.md).

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements-equipo.txt
python scripts/descargar_idsse.py         # 7 partidos de Bundesliga, ~2,6 GB, a data/external/
python -m pytest -q tests                 # 7 tests; los del reto se saltan si no tienes data/
python scripts/evaluar_externo.py         # baselines en IDSSE degradado (~2 min)
python scripts/generar_entrega.py baseline_smooth50   # CSV validado + sha256 en salidas/
```

IDSSE es nuestro conjunto de validación: partidos de **otros** equipos con, en cada frame, balón
vivo/muerto y equipo en posesión. Los organizadores aprobaron su uso (issue #16). Es CC BY 4.0, así
que hay que citarlo en el documento técnico y en el README de entrega. Como sus datos son públicos,
con IDSSE sí puedes mirar filas, enseñárselas a un LLM o subirlas a MareNostrum.

Si tienes varios checkouts o worktrees, apunta todos a la misma carpeta para no bajar 2,6 GB cada
vez:

```bash
export MTAH_EXTERNAL_DIR=/ruta/compartida/external
```

## Reglas que afectan a los datos

Saltarse cualquiera de estas reglas puede suponer la descalificación, no una penalización. Los
organizadores pueden volver a ejecutar el repo ganador.

- **Nada de `data/` se commitea.** El `.gitignore` ya lo excluye; no lo fuerces con `git add -f`.
  Tampoco salidas derivadas que reconstruyan los datos (exportaciones de tracking por frame, clips
  de vídeo).
- **No subas datos a LLMs alojados** (ChatGPT, Claude, Gemini…), ni ficheros ni trozos copiados y
  pegados: ni filas del ATD, ni capturas del vídeo, ni transcripciones. A un agente puedes pedirle
  código y enseñarle métricas agregadas, no contenido del partido. Los modelos locales sí están
  permitidos.
- **No etiquetes este partido a mano**, ni para la entrega ni para entrenar o validar. Mirar el
  vídeo para entender los datos sí está permitido. Etiquetar otros partidos también.
- **El GT público (5 minutos) es solo para comprobar que el pipeline está bien alineado**, no para
  elegir entre versiones ni para ajustar parámetros.
- **MareNostrum:** de momento, allí solo van datos de otros partidos. No copies los datos del reto
  al clúster sin hablarlo antes.
