# Méthodologie Expérimentale

## Configuration Expérimentale

### Dataset
- **Nom:** Weeds-3 Agricultural Object Detection
- **Images d'entraînement:** 3,664
- **Images de validation:** 359
- **Domaine:** Détection de mauvaises herbes agricoles

### Modèles Testés
- **YOLOv12-N:** Modèle baseline avec Area Attention
- **YOLOv13-N:** Modèle avec HyperACE (Hypergraph attention)

### Mécanismes d'Attention
- **SDPA:** Scaled Dot-Product Attention (PyTorch natif)
- **Flash Attention:** Version 2.7.3 avec optimisations mémoire

### Paramètres d'Entraînement
- **Epochs:** 20 (mode développement)
- **Batch Size:** 8 (optimisé RTX 4060)
- **Optimizer:** AdamW
- **Learning Rate:** 0.001 avec cosine decay
- **Image Size:** 640×640

## Métriques Évaluées

- **mAP50:** Mean Average Precision à IoU=0.5
- **mAP50-95:** Mean Average Precision à IoU=0.5:0.95
- **Précision:** Taux de vrais positifs
- **Rappel:** Taux de détection des objets réels
- **Temps d'entraînement:** Durée totale en minutes
- **Utilisation mémoire:** GPU et CPU

## Protocole de Validation

1. **Reproductibilité:** Seeds fixes, environnement contrôlé
2. **Équité:** Paramètres identiques entre configurations
3. **Validation croisée:** Multiple runs sur mêmes données
4. **Analyse statistique:** Tests de significativité
