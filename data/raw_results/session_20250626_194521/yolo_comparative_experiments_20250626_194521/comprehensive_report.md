# 🚀 RAPPORT EXPÉRIENCES COMPARATIVES YOLO

- **CPU:** AMD Ryzen 9 7945HX (12 cores)
- **RAM:** 39 GB disponible
- **Stockage:** 899 GB libre
- **OS:** Linux (WSL2)
- **Environnement:** flash-attention conda env

- **Session ID:** 20250626_194521
- **Durée totale:** 123.0 minutes
- **Date:** 2025-06-26 21:48:20
- **Dataset:** /mnt/c/Users/kitok/Desktop/SmartFarm_OS/IA/YOLO12_FLASH_attn/Weeds-3
- **Chercheur:** Kennedy Kitoko 🇨🇩

## Résultats Expériences
- **Total:** 4
- **Réussies:** 2
- **Échouées:** 2

### Détails par Expérience

#### YOLOV12_SDPA - ✅ Réussie
- **Durée:** 55.3 minutes
- **mAP50:** 0.767
- **mAP50-95:** 0.461
- **Précision:** 0.816
- **Rappel:** 0.664
- **Mémoire utilisée:** 8394.3 MB
- **Mémoire GPU:** 0.22 GB
- **Résultats:** yolo_comparative_experiments_20250626_194521/yolov12_sdpa

#### YOLOV12_FLASH - ✅ Réussie
- **Durée:** 67.3 minutes
- **mAP50:** 0.765
- **mAP50-95:** 0.479
- **Précision:** 0.831
- **Rappel:** 0.632
- **Mémoire utilisée:** 3406.0 MB
- **Mémoire GPU:** 0.05 GB
- **Résultats:** yolo_comparative_experiments_20250626_194521/yolov12_flash

#### YOLOV13_SDPA - ❌ Échouée
- **Erreur:** [Errno 2] No such file or directory: 'yolo13n.pt'

#### YOLOV13_FLASH - ❌ Échouée
- **Erreur:** [Errno 2] No such file or directory: 'yolo13n.pt'

## Analyse Comparative

### YOLOv12 vs YOLOv13
- **Gagnant:** N/A
- **Différence mAP50:** 0.000
- **Différence vitesse:** 0.0 min

### SDPA vs Flash Attention
- **Gagnant:** SDPA
- **Différence mAP50:** 0.002
- **Différence vitesse:** 12.0 min

### Classement Performance
1. **yolov12_sdpa:** mAP50=0.767, Durée=55.3min
2. **yolov12_flash:** mAP50=0.765, Durée=67.3min

### Recommandations
- Meilleure configuration: yolov12_sdpa
- mAP50 optimale: 0.767
- Durée d'entraînement: 55.3 minutes

## Innovations Techniques

### YOLOv13 - Innovations
- **HyperACE:** Mécanisme d'attention basé sur les hypergraphes
- **FullPAD:** Paradigme de distribution complète des caractéristiques
- **DS-Blocks:** Blocs de convolution séparable en profondeur

### Flash Attention vs SDPA
- **SDPA:** Attention native PyTorch optimisée
- **Flash Attention:** Optimisation mémoire avec réordonnancement
- **Complexité:** Linéaire vs quadratique pour les longues séquences

---
**Développé par Kennedy Kitoko 🇨🇩**
*Démocratisation de l'IA pour l'Agriculture Mondiale*
