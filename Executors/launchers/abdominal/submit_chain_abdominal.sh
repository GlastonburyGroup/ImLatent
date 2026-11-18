#!/bin/bash
# Submits the two-job chain used for the abdominal organ-specific 2D DiffAE models (F20204 / F20254 / F20259):
#   job 1: train + validate (early stopping) + test on the fold's test split   (run_mode 4, default)
#   job 2: whole-dataset embedding extraction from the best checkpoint      (waits for job 1)
# Per-dataset/organ defaults (input shape, batch size, channels, data JSON) match the original abdominal runs.
#
# usage: submit_chain_abdominal.sh --dataset 20204|20254|20259 [--data_type crop|full] [--wnbgroup Liver|Spleen|Aorta|Pancreas|Kidney]
#                                  [--latent_dim 128] [--seed 1701] [--trainID init_2D_128] [--test] [--dependency JOBID] ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 1. Defaults ---
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROGRAMME="Executors/main_diffAE2D.py"
WRAPPER="$SCRIPT_DIR/gpurun_uv.sh"

DATASET=""
LATENT_DIM="128"
NET_CH="32"
SAMPLE_SIZE="4"
TRAIN_ID=""
SEED="1701"
PATIENCE="5"

DATA_PATH=""
DATA_TYPE="crop"   # crop, full
WNB_GROUP=""

IN_CHANNELS=""
OUT_CHANNELS=""
INPUT_SHAPE=""
BATCH_SIZE=""

NO_TBACTIVE="--no-tbactive"
EXTRA_ARGS=""
JOB2_EXTRAS="--json§foldCSV 0 --resume --load_best --run_mode 2 --output_suffix fullDSV3 --test_emb_only"
TEST_ONLY=false
EXTERNAL_DEP=""

# --- 2. Arguments ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)        DATASET="$2"; shift 2 ;;
    --root)           ROOT_DIR="$2"; shift 2 ;;
    --programme)      PROGRAMME="$2"; shift 2 ;;
    --latent_dim)     LATENT_DIM="$2"; shift 2 ;;
    --net_ch)         NET_CH="$2"; shift 2 ;;
    --sample_size)    SAMPLE_SIZE="$2"; shift 2 ;;
    --trainID)        TRAIN_ID="$2"; shift 2 ;;
    --seed)           SEED="$2"; shift 2 ;;
    --patience)       PATIENCE="$2"; shift 2 ;;
    --datajson)       DATA_PATH="$2"; shift 2 ;;
    --data_type)      DATA_TYPE="$2"; shift 2 ;;
    --wnbgroup)       WNB_GROUP="$2"; shift 2 ;;
    --in_channels)    IN_CHANNELS="$2"; shift 2 ;;
    --out_channels)   OUT_CHANNELS="$2"; shift 2 ;;
    --input_shape)    INPUT_SHAPE="$2"; shift 2 ;;
    --batch_size)     BATCH_SIZE="$2"; shift 2 ;;
    --extra_args)     EXTRA_ARGS="$2"; shift 2 ;;   # any further arguments for the python script, e.g. "--json§datadir /my/data --wnbentity me"
    --with-tbactive)  NO_TBACTIVE=""; shift ;;
    --test)           TEST_ONLY=true; shift ;;
    --dependency)     EXTERNAL_DEP="$2"; shift 2 ;;
    *) echo "Error: Unknown argument '$1'"; exit 1 ;;
  esac
done

# trainID rule of the original runs: seed 1701 -> init_2D_<dim>, any other seed -> init_2D_<seed>_<dim>
if [ -z "$TRAIN_ID" ]; then
    if [ "$SEED" == "1701" ]; then TRAIN_ID="init_2D_${LATENT_DIM}"; else TRAIN_ID="init_2D_${SEED}_${LATENT_DIM}"; fi
fi

# --- 3. Dataset- and organ-specific defaults (as in the original runs) ---
case $DATASET in
  20204)  # F20204 Liver ShMoLLI, FIT PARAMS (7 channels)
    [ -z "$WNB_GROUP" ] && WNB_GROUP="Liver"
    CH="7"; BASE_PATH="Configs/UKBB204/dataNpath_F20204_FITPARAMS"; FULL_JSON="${BASE_PATH}_HPC.json"; CROP_SUFFIX="_HPC.json"
    FULL_SHAPE="288,384"; FULL_BS="24"
    case ${WNB_GROUP,,} in
      liver)  CROP_SHAPE="224,256"; CROP_BS="54" ;;
      spleen) CROP_SHAPE="128,128"; CROP_BS="54" ;;
      aorta)  CROP_SHAPE="64,64";   CROP_BS="54" ;;
      *) CROP_SHAPE=""; CROP_BS="" ;;
    esac ;;
  20254)  # F20254 Liver IDEAL (6 channels)
    [ -z "$WNB_GROUP" ] && WNB_GROUP="Liver"
    CH="6"; BASE_PATH="Configs/UKBB254/dataNpath_F20254"; FULL_JSON="${BASE_PATH}_HPCBerry.json"; CROP_SUFFIX="_HPCBerry.json"
    FULL_SHAPE="256,256"; FULL_BS="28"
    case ${WNB_GROUP,,} in
      liver)  CROP_SHAPE="192,192"; CROP_BS="54" ;;
      spleen) CROP_SHAPE="96,96";   CROP_BS="54" ;;
      aorta)  CROP_SHAPE="32,32";   CROP_BS="54" ;;
      *) CROP_SHAPE=""; CROP_BS="" ;;
    esac ;;
  20259)  # F20259 Pancreas ShMoLLI, FIT PARAMS (7 channels)
    [ -z "$WNB_GROUP" ] && WNB_GROUP="Pancreas"
    CH="7"; BASE_PATH="Configs/UKBB259/dataNpath_F20259_FITPARAMS"; FULL_JSON="${BASE_PATH}_HPC.json"; CROP_SUFFIX="_HPC.json"
    FULL_SHAPE="288,384"; FULL_BS="24"   # never validated by a finished run: check the image size of the F20259 HDF5
    case ${WNB_GROUP,,} in
      pancreas) CROP_SHAPE="96,160";  CROP_BS="40" ;;
      spleen)   CROP_SHAPE="128,128"; CROP_BS="54" ;;
      aorta)    CROP_SHAPE="32,32";   CROP_BS="54" ;;
      kidney)   CROP_SHAPE="192,320"; CROP_BS="24" ;;
      *) CROP_SHAPE=""; CROP_BS="" ;;
    esac ;;
  *) echo "Error: --dataset must be one of 20204, 20254, 20259"; exit 1 ;;
esac

[ -z "$IN_CHANNELS" ]  && IN_CHANNELS="$CH"
[ -z "$OUT_CHANNELS" ] && OUT_CHANNELS="$CH"

if [ "$DATA_TYPE" == "full" ]; then
    [ -z "$INPUT_SHAPE" ] && INPUT_SHAPE="$FULL_SHAPE"
    [ -z "$BATCH_SIZE" ]  && BATCH_SIZE="$FULL_BS"
    [ -z "$DATA_PATH" ]   && DATA_PATH="$FULL_JSON"
else
    [ -z "$INPUT_SHAPE" ] && INPUT_SHAPE="$CROP_SHAPE"
    [ -z "$BATCH_SIZE" ]  && BATCH_SIZE="$CROP_BS"
    [ -z "$DATA_PATH" ]   && DATA_PATH="${BASE_PATH}_${WNB_GROUP,,}Crop${CROP_SUFFIX}"
fi

if [ -z "$INPUT_SHAPE" ] || [ -z "$BATCH_SIZE" ]; then
    echo "Error: no defaults for organ '$WNB_GROUP' in dataset $DATASET; pass --input_shape and --batch_size."
    exit 1
fi
if [ ! -f "$ROOT_DIR/$DATA_PATH" ]; then
    echo "Error: data JSON not found: $ROOT_DIR/$DATA_PATH"
    exit 1
fi

# --- 4. Arguments for the python script ---
BASE_ARGS="--model§latent_dim $LATENT_DIM --model§DiffAE§net_ch $NET_CH --model§DiffAE§sample_size $SAMPLE_SIZE \
--trainID $TRAIN_ID --seed $SEED --early_stopping_patience $PATIENCE \
--datajson_path $DATA_PATH --wnbgroup $WNB_GROUP \
--in_channels $IN_CHANNELS --out_channels $OUT_CHANNELS --input_shape $INPUT_SHAPE \
$NO_TBACTIVE --batch_size $BATCH_SIZE $EXTRA_ARGS"
BASE_ARGS=$(echo "$BASE_ARGS" | xargs)

GROUP_STR="${WNB_GROUP:0:3}"; TYPE_STR="${DATA_TYPE^}"
JOB_NAME_1="dAE${DATASET}${GROUP_STR}${TYPE_STR}"
JOB_NAME_2="dAET${DATASET}${GROUP_STR}${TYPE_STR}"

echo "------------------------------------------------"
echo "Dataset:   F$DATASET  Organ: $WNB_GROUP ($DATA_TYPE)"
echo "JSON:      $DATA_PATH"
echo "Config:    Shape=$INPUT_SHAPE, Batch=$BATCH_SIZE, Channels=$IN_CHANNELS, z=$LATENT_DIM, seed=$SEED, trainID=$TRAIN_ID"
echo "Args:      $BASE_ARGS"
echo "------------------------------------------------"

# --- 5. Submit ---
DEP_ARGS=()
[ -n "$EXTERNAL_DEP" ] && DEP_ARGS=(--dependency="afterok:$EXTERNAL_DEP")

if [ "$TEST_ONLY" = false ]; then
    JOB_ID_1=$(sbatch --parsable "${DEP_ARGS[@]}" -J "$JOB_NAME_1" "$WRAPPER" --root "$ROOT_DIR" --programme "$PROGRAMME" --args "$BASE_ARGS")
    [ -z "$JOB_ID_1" ] && { echo "Error: Job 1 failed to submit."; exit 1; }
    echo "Submitted Job 1: $JOB_ID_1"
    JOB_ID_2=$(sbatch --parsable --dependency=afterok:$JOB_ID_1 -J "$JOB_NAME_2" "$WRAPPER" --root "$ROOT_DIR" --programme "$PROGRAMME" --args "$BASE_ARGS $JOB2_EXTRAS")
    [ -z "$JOB_ID_2" ] && { echo "Error: Job 2 failed to submit."; exit 1; }
    echo "Submitted Job 2: $JOB_ID_2  <-- waits for Job $JOB_ID_1"
else
    echo "[--test mode] Skipping Job 1, submitting Job 2 only."
    JOB_ID_2=$(sbatch --parsable "${DEP_ARGS[@]}" -J "$JOB_NAME_2" "$WRAPPER" --root "$ROOT_DIR" --programme "$PROGRAMME" --args "$BASE_ARGS $JOB2_EXTRAS")
    [ -z "$JOB_ID_2" ] && { echo "Error: Job 2 failed to submit."; exit 1; }
    echo "Submitted Job 2: $JOB_ID_2"
fi
