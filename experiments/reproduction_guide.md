# 🔬 Guide de Reproduction - YOLOv12 vs YOLOv13 Attention Study

**Auteur:** Kennedy Kitoko 🇨🇩  
**Date:** 27 Juin 2025  
**Version:** 1.0

## 📋 Prérequis

### Matériel Recommandé
- **GPU:** NVIDIA RTX 4060 ou supérieur (8GB+ VRAM)
- **CPU:** AMD Ryzen 9 7945HX ou équivalent
- **RAM:** 32GB minimum (39GB recommandé)
- **Stockage:** 50GB d'espace libre

### Système d'Exploitation
- **Linux:** Ubuntu 20.04+ (recommandé)
- **Windows:** WSL2 avec Ubuntu
- **macOS:** M1/M2 avec Rosetta 2

## 🚀 Installation de l'Environnement

### 1. Cloner le Repository
```bash
git clone https://github.com/kennedy-kitoko/yolov12-vs-yolov13-attention-study.git
cd yolov12-vs-yolov13-attention-study
```

### 2. Créer l'Environnement Conda
```bash
# Créer l'environnement
conda env create -f environment.yml

# Activer l'environnement
conda activate flash-attention

# Vérifier l'installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"
```

### 3. Installer les Dépendances Supplémentaires
```bash
pip install -r requirements.txt
```

## 📊 Préparation des Données

### 1. Télécharger le Dataset Weeds-3
```bash
# Créer le dossier dataset
mkdir -p Weeds-3

# Télécharger depuis Roboflow (exemple)
# wget https://app.roboflow.com/ds/weeds-3-dataset.zip
# unzip weeds-3-dataset.zip -d Weeds-3/
```

### 2. Vérifier la Structure des Données
```
Weeds-3/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

### 3. Vérifier le fichier data.yaml
```yaml
path: ../Weeds-3
train: train/images
val: valid/images

nc: 1  # nombre de classes
names: ['weed']  # noms des classes
```

## 🔬 Exécution des Expériences

### 1. Validation de l'Environnement
```bash
python src/comprehensive_yolo_experiments.py --validate-only
```

### 2. Exécution Complète (Toutes les Expériences)
```bash
python src/comprehensive_yolo_experiments.py --epochs 20 --batch-size 8
```

### 3. Exécution d'Expériences Spécifiques
```bash
# YOLOv12 + SDPA seulement
python src/comprehensive_yolo_experiments.py --experiments yolov12_sdpa

# YOLOv13 + Flash Attention seulement
python src/comprehensive_yolo_experiments.py --experiments yolov13_flash
```

### 4. Mode Développement (Tests Rapides)
```bash
python src/comprehensive_yolo_experiments.py --epochs 5 --batch-size 4 --dev-mode
```

## 📈 Analyse des Résultats

### 1. Analyse Automatique
```bash
python src/data_analysis.py
```

### 2. Génération des Visualisations
```bash
python src/visualization.py
```

### 3. Rapport Complet
```bash
# Génère analysis_report.md et consolidated_results.csv
python src/data_analysis.py --generate-report
```

## 📁 Structure des Résultats

Après exécution, vous trouverez :

```
yolov12-vs-yolov13-attention-study/
├── data/
│   ├── raw_results/
│   │   ├── session_YYYYMMDD_HHMMSS/
│   │   │   ├── yolov12_sdpa/
│   │   │   ├── yolov12_flash/
│   │   │   ├── yolov13_sdpa/
│   │   │   └── yolov13_flash/
│   ├── processed/
│   │   ├── results_yolov12_sdpa.csv
│   │   ├── results_yolov12_flash.csv
│   │   ├── results_yolov13_sdpa.csv
│   │   └── results_yolov13_flash.csv
│   └── consolidated_results.json
├── figures/
│   ├── mAP50_comparison.png
│   ├── performance_radar.png
│   ├── efficiency_analysis.png
│   ├── memory_analysis.png
│   └── correlation_heatmap.png
└── analysis_report.md
```

## 🔧 Dépannage

### Problèmes Courants

#### 1. Erreur CUDA
```bash
# Vérifier l'installation CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

#### 2. Erreur Flash Attention
```bash
# Réinstaller Flash Attention
pip uninstall flash-attn
pip install flash-attn --no-build-isolation
```

#### 3. Erreur Mémoire GPU
```bash
# Réduire la taille du batch
python src/comprehensive_yolo_experiments.py --batch-size 4
```

#### 4. Erreur Dataset
```bash
# Vérifier le chemin du dataset
ls -la Weeds-3/
cat Weeds-3/data.yaml
```

## 📊 Validation des Résultats

### Métriques Attendues (Approximatives)
| Configuration | mAP50 (%) | mAP50-95 (%) | Temps (min) |
|---------------|-----------|--------------|-------------|
| YOLOv12 + SDPA | 76.7 | 46.1 | 55.3 |
| YOLOv12 + Flash | 76.5 | 47.9 | 67.3 |
| YOLOv13 + SDPA | 82.9 | 47.4 | 58.6 |
| YOLOv13 + Flash | 82.3 | 52.3 | 65.7 |

### Tolérance Acceptable
- **mAP50:** ±2%
- **Temps d'entraînement:** ±10%
- **Mémoire GPU:** ±20%

## 📚 Ressources Supplémentaires

### Documentation
- `docs/methodology.md` - Méthodologie détaillée
- `docs/results_interpretation.md` - Interprétation des résultats
- `experiments/experiment_config.yaml` - Configuration complète

### Notebooks Jupyter
- `notebooks/01_data_exploration.ipynb` - Exploration des données
- `notebooks/02_results_analysis.ipynb` - Analyse des résultats
- `notebooks/03_visualization.ipynb` - Génération de graphiques

## 🤝 Support

Pour toute question ou problème :
- **Issues GitHub:** [Repository Issues](https://github.com/kennedy-kitoko/yolov12-vs-yolov13-attention-study/issues)
- **Email:** kennedy.kitoko@agricultural-ai.org
- **Documentation:** Consulter les fichiers dans `docs/`

---

**Note:** Ce guide garantit la reproductibilité complète des expériences. Tous les paramètres et configurations sont documentés dans `experiment_config.yaml`. 