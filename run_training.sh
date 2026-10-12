#!/usr/bin/env bash
# Pipeline de entrenamiento local (correr desde la raíz del repo, después de src/prepare_data.py).
# GPU: un trabajo a la vez (4 GB no alcanzan para dos). CPU: los modelos sobre DINOv2 corren en
# paralelo con el entrenamiento de las CNN. Cada modelo escribe sus propios archivos en results/.
set -euo pipefail
PY=${PY:-C:/Users/Acer/miniforge3/envs/mosquito-ml/python.exe}
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
mkdir -p logs

# Pasos ya completados se omiten (permite reanudar sin repetir trabajo)
[ -f data/processed/dinov2_feats.npz ] || $PY src/extract_features.py 2>&1 | tee logs/extract_features.log

CPU_PID=""
if [ ! -f results/metrics_dinov2_hier_cascade.json ]; then
  $PY src/train_feature_models.py --jobs 2 > logs/feature_models.log 2>&1 &
  CPU_PID=$!
fi

[ -f results/metrics_efficientnet_b0.json ] || $PY src/train_cnn.py --arch efficientnet_b0 --epochs 6 --batch 32 --lr 5e-4 2>&1 | tee logs/efficientnet_b0.log
# ConvNeXt-Tiny: 13 img/s a 224 px en la GTX 1650 -> 176 px con stem + 2 primeras etapas congeladas
# Batch 32: con batch 64 EfficientNet usaba 3.7 de 4 GB y el driver de Windows desbordaba la VRAM a RAM (38 vs 120 img/s)
[ -f results/metrics_convnext_tiny.json ] || $PY src/train_cnn.py --arch convnext_tiny --epochs 3 --batch 32 --lr 3e-4 --img-size 176 --freeze-stages 2 2>&1 | tee logs/convnext_tiny.log

# Detector local (alternativa a Colab). Si ya llegaron resultados de Colab, se omite.
if [ ! -f models/detector/det_test_boxes.csv ]; then
  $PY src/train_detector.py --imgsz 512 --epochs 12 --batch 16 --workers 3 --hours 0.3 2>&1 | tee logs/detector.log
fi

[ -z "$CPU_PID" ] || wait $CPU_PID
$PY src/summarize_results.py 2>&1 | tee logs/summary.log
echo "Entrenamiento completo"
