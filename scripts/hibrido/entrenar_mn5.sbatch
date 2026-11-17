#!/bin/bash
# Entrena el modelo de emisiones en una H100 de MareNostrum 5, solo con IDSSE (datos externos).
# Nada del partido del reto va al clúster: la predicción de BAR-ATH se hace en local con final.pt.
#
#   rsync -a --delete mtah scripts marenostrum:mtah2027/code/
#   rsync -a $MTAH_EXTERNAL_DIR/cache/idsse_*.npz marenostrum:mtah2027/external/cache/
#   sbatch --output=$HOME/mtah2027/logs/%x-%j.out \
#     --export=ALL,RUNS="it2:--final" scripts/hibrido/entrenar_mn5.sbatch
#
# RUNS: lista de "nombre:argumentos extra" separados por ';', se ejecutan en serie.
# Ojo: --export separa por comas; dentro de RUNS usa "+" en listas (--fuentes idsse+metrica).
#SBATCH --job-name=mtah-hibrido
# La cuenta va en la línea de comandos: sbatch --account=<cuenta> ...
#SBATCH --qos=acc_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
set -euo pipefail
: "${RUNS:?}"
BASE=$HOME/mtah2027
SIF=${MTAH_SIF:?ruta a una imagen Singularity con Python 3.12 y torch}
module load singularity/4.1.5 >/dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
py() {
  singularity exec --nv --bind /gpfs --pwd "$BASE/code" \
    --env "PYTHONPATH=$BASE/code,MTAH_EXTERNAL_DIR=$BASE/external,OMP_NUM_THREADS=1" "$SIF" python "$@"
}
py -c "import sys, numpy, torch; print(sys.version.split()[0], numpy.__version__, torch.__version__)"
IFS=';' read -ra LISTA <<< "$RUNS"
for item in "${LISTA[@]}"; do
  nombre=${item%%:*}; extra=${item#*:}
  echo "=== run $nombre ($extra)"
  # shellcheck disable=SC2086
  py scripts/hibrido/entrenar.py --run "$nombre" --salida "$BASE/runs" --procesos 18 $extra
done
