# 🚀 RAPPORT EXPÉRIENCES COMPARATIVES YOLO

- **CPU:** AMD Ryzen 9 7945HX (12 cores)
- **RAM:** 39 GB disponible
- **Stockage:** 899 GB libre
- **OS:** Linux (WSL2)
- **Environnement:** flash-attention conda env

- **Session ID:** 20250627_012822
- **Durée totale:** 65.7 minutes
- **Date:** 2025-06-27 02:34:06
- **Dataset:** /mnt/c/Users/kitok/Desktop/SmartFarm_OS/IA/YOLO12_FLASH_attn/Weeds-3
- **Chercheur:** Kennedy Kitoko 🇨🇩

## Résultats Expériences
- **Total:** 4
- **Réussies:** 1
- **Échouées:** 3

### Détails par Expérience

#### YOLOV12_SDPA - ❌ Échouée
- **Erreur:** Inconnue

#### YOLOV12_FLASH - ❌ Échouée
- **Erreur:** Inconnue

#### YOLOV13_SDPA - ❌ Échouée
- **Erreur:** Inconnue

#### YOLOV13_FLASH - ✅ Réussie
- **Durée:** 65.7 minutes
- **mAP50:** 0.823
- **mAP50-95:** 0.523
- **Précision:** 0.894
- **Rappel:** 0.684
- **Mémoire utilisée:** 8364.4 MB
- **Mémoire GPU:** 0.25 GB
- **Résultats:** yolo_comparative_experiments_20250627_012822/yolov13_flash

## Analyse Comparative

### YOLOv12 vs YOLOv13
- **Gagnant:** N/A
- **Différence mAP50:** 0.000
- **Différence vitesse:** 0.0 min

### SDPA vs Flash Attention
- **Gagnant:** N/A
- **Différence mAP50:** 0.000
- **Différence vitesse:** 0.0 min

### Classement Performance
1. **yolov13_flash:** mAP50=0.823, Durée=65.7min

### Recommandations
- Meilleure configuration: yolov13_flash
- mAP50 optimale: 0.823
- Durée d'entraînement: 65.7 minutes

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
