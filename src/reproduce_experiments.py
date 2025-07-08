#!/usr/bin/env python3
"""
🔄 SCRIPT DE REPRODUCTION DES EXPÉRIENCES
YOLOv12 vs YOLOv13 - SDPA vs Flash Attention Study
Développé par Kennedy Kitoko 🇨🇩
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

class ExperimentReproducer:
    """Reproducteur d'expériences YOLO avec validation complète"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.configs = self._load_experiment_configs()
        
    def _load_experiment_configs(self) -> Dict[str, Any]:
        """Charge les configurations d'expériences validées"""
        return {
            'yolov12_sdpa': {
                'model': 'yolo12n.pt',
                'attention': 'sdpa',
                'expected_map50': 76.72,
                'expected_duration_min': 55.3,
                'epochs': 20,
                'batch': 8,
                'description': 'YOLOv12 with native PyTorch SDPA'
            },
            'yolov12_flash': {
                'model': 'yolo12n.pt', 
                'attention': 'flash',
                'expected_map50': 76.53,
                'expected_duration_min': 67.3,
                'epochs': 20,
                'batch': 8,
                'description': 'YOLOv12 with Flash Attention 2.7.3'
            },
            'yolov13_sdpa': {
                'model': 'yolov13n.pt',
                'attention': 'sdpa',
                'expected_map50': 82.9,
                'expected_duration_min': 58.6,
                'epochs': 20,
                'batch': 8,
                'description': 'YOLOv13 with HyperACE + SDPA'
            },
            'yolov13_flash': {
                'model': 'yolov13n.pt',
                'attention': 'flash', 
                'expected_map50': 82.3,
                'expected_duration_min': 65.7,
                'epochs': 20,
                'batch': 8,
                'description': 'YOLOv13 with HyperACE + Flash Attention'
            }
        }
    
    def validate_environment(self) -> bool:
        """Validation complète de l'environnement de reproduction"""
        print("🔍 VALIDATION ENVIRONNEMENT DE REPRODUCTION")
        print("=" * 50)
        
        checks = {
            'python': False,
            'torch': False,
            'cuda': False,
            'ultralytics': False,
            'flash_attn': False,
            'dataset': False,
            'weights': False
        }
        
        # Python version
        if sys.version_info >= (3, 8):
            checks['python'] = True
            print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
        else:
            print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} (requis: ≥3.8)")
        
        # PyTorch + CUDA
        try:
            import torch
            checks['torch'] = True
            print(f"✅ PyTorch {torch.__version__}")
            
            if torch.cuda.is_available():
                checks['cuda'] = True
                print(f"✅ CUDA {torch.version.cuda} - {torch.cuda.get_device_name(0)}")
            else:
                print("❌ CUDA non disponible")
        except ImportError:
            print("❌ PyTorch non installé")
        
        # Ultralytics
        try:
            from ultralytics import YOLO
            checks['ultralytics'] = True
            print("✅ Ultralytics YOLO")
        except ImportError:
            print("❌ Ultralytics non installé")
        
        # Flash Attention
        try:
            import flash_attn
            checks['flash_attn'] = True
            print(f"✅ Flash Attention {getattr(flash_attn, '__version__', 'unknown')}")
        except ImportError:
            print("⚠️ Flash Attention non disponible (optionnel)")
        
        # Dataset Weeds-3
        dataset_paths = [
            'Weeds-3',
            '/mnt/c/Users/kitok/Desktop/SmartFarm_OS/IA/YOLO12_FLASH_attn/Weeds-3',
            '../Weeds-3'
        ]
        
        for path in dataset_paths:
            if os.path.exists(os.path.join(path, 'train', 'images')):
                checks['dataset'] = True
                print(f"✅ Dataset: {os.path.abspath(path)}")
                self.dataset_path = os.path.abspath(path)
                break
        
        if not checks['dataset']:
            print("❌ Dataset Weeds-3 non trouvé")
        
        # Poids modèles
        weights_available = []
        for config_name, config in self.configs.items():
            model_file = config['model']
            if os.path.exists(model_file):
                weights_available.append(model_file)
        
        if len(weights_available) >= 2:
            checks['weights'] = True
            print(f"✅ Poids modèles: {', '.join(weights_available)}")
        else:
            print(f"⚠️ Poids partiels: {', '.join(weights_available)}")
        
        # Résumé validation
        critical_checks = ['python', 'torch', 'cuda', 'ultralytics', 'dataset']
        critical_passed = sum(checks[check] for check in critical_checks)
        
        ready = critical_passed >= 5
        
        print(f"\n📊 Validation: {sum(checks.values())}/{len(checks)} checks")
        print(f"🎯 Critique: {critical_passed}/{len(critical_checks)} essentiels")
        print(f"🏁 Prêt pour reproduction: {'Oui' if ready else 'Non'}")
        
        return ready
    
    def reproduce_single_experiment(self, config_name: str, 
                                  epochs: Optional[int] = None,
                                  batch: Optional[int] = None,
                                  quick_mode: bool = False) -> Dict[str, Any]:
        """Reproduit une expérience spécifique"""
        
        if config_name not in self.configs:
            raise ValueError(f"Configuration inconnue: {config_name}")
        
        config = self.configs[config_name].copy()
        
        # Override paramètres si spécifiés
        if epochs:
            config['epochs'] = epochs
        if batch:
            config['batch'] = batch
        if quick_mode:
            config['epochs'] = min(config['epochs'], 5)
        
        print(f"\n🔬 REPRODUCTION: {config_name.upper()}")
        print(f"📋 {config['description']}")
        print("=" * 50)
        
        # Import du script principal
        try:
            from comprehensive_yolo_experiments import ComprehensiveYOLOExperimentLauncher
        except ImportError:
            print("❌ Script principal non trouvé")
            return {'success': False, 'error': 'Script principal manquant'}
        
        # Configuration launcher
        launcher = ComprehensiveYOLOExperimentLauncher(self.dataset_path)
        
        # Extraction modèle et attention
        model_version = 'yolov12' if 'yolov12' in config_name else 'yolov13'
        attention_type = config['attention']
        
        print(f"🎯 Modèle: {model_version}")
        print(f"⚡ Attention: {attention_type}")
        print(f"📊 Époques: {config['epochs']}")
        print(f"🔢 Batch: {config['batch']}")
        
        # Lancement expérience
        try:
            start_time = datetime.now()
            
            result = launcher.run_yolo_experiment(
                model_version=model_version,
                attention_type=attention_type,
                epochs=config['epochs'],
                batch_size=config['batch']
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            
            # Validation résultats
            if result['success']:
                actual_map50 = result['metrics']['mAP50'] * 100
                expected_map50 = config['expected_map50']
                map50_diff = abs(actual_map50 - expected_map50)
                
                print(f"\n📈 RÉSULTATS REPRODUCTION:")
                print(f"   mAP50: {actual_map50:.2f}% (attendu: {expected_map50:.2f}%)")
                print(f"   Différence: {map50_diff:.2f}%")
                print(f"   Durée: {duration:.1f} min (attendu: {config['expected_duration_min']:.1f} min)")
                
                # Tolérance validation
                tolerance = 3.0 if quick_mode else 1.5  # % de tolérance
                validated = map50_diff <= tolerance
                
                result['reproduction_validation'] = {
                    'validated': validated,
                    'actual_map50': actual_map50,
                    'expected_map50': expected_map50,
                    'map50_difference': map50_diff,
                    'tolerance': tolerance,
                    'duration_actual': duration,
                    'duration_expected': config['expected_duration_min']
                }
                
                if validated:
                    print("✅ Reproduction validée (dans tolérance)")
                else:
                    print(f"⚠️ Reproduction hors tolérance (>{tolerance}%)")
            
            return result
            
        except Exception as e:
            print(f"❌ Erreur reproduction: {e}")
            return {
                'success': False,
                'error': str(e),
                'config': config_name
            }
    
    def reproduce_all_experiments(self, epochs: Optional[int] = None,
                                batch: Optional[int] = None,
                                quick_mode: bool = False) -> Dict[str, Any]:
        """Reproduit toutes les expériences"""
        print("🚀 REPRODUCTION COMPLÈTE - TOUTES EXPÉRIENCES")
        print("🇨🇩 Kennedy Kitoko - Agricultural AI Innovation")
        print("=" * 60)
        
        results = {}
        successful_reproductions = 0
        validated_reproductions = 0
        
        for config_name in self.configs.keys():
            try:
                print(f"\n{'='*20}")
                print(f"EXPÉRIENCE {len(results)+1}/4: {config_name.upper()}")
                print(f"{'='*20}")
                
                result = self.reproduce_single_experiment(
                    config_name, epochs, batch, quick_mode
                )
                
                results[config_name] = result
                
                if result['success']:
                    successful_reproductions += 1
                    if result.get('reproduction_validation', {}).get('validated', False):
                        validated_reproductions += 1
                
                # Pause entre expériences
                if len(results) < len(self.configs):
                    print("⏸️ Pause nettoyage mémoire...")
                    try:
                        import torch
                        import gc
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        import time
                        time.sleep(5)
                    except:
                        pass
                        
            except Exception as e:
                print(f"❌ Erreur expérience {config_name}: {e}")
                results[config_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Résumé final
        print(f"\n{'='*30}")
        print("🎉 REPRODUCTION TERMINÉE")
        print(f"{'='*30}")
        
        print(f"📊 Expériences réussies: {successful_reproductions}/{len(self.configs)}")
        print(f"✅ Reproductions validées: {validated_reproductions}/{successful_reproductions}")
        
        # Sauvegarde résultats
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"reproduction_results_{timestamp}.json"
        
        reproduction_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.configs),
            'successful_reproductions': successful_reproductions,
            'validated_reproductions': validated_reproductions,
            'parameters': {
                'epochs': epochs,
                'batch': batch,
                'quick_mode': quick_mode
            },
            'results': results
        }
        
        with open(results_file, 'w') as f:
            json.dump(reproduction_summary, f, indent=4, default=str)
        
        print(f"💾 Résultats sauvegardés: {results_file}")
        
        return reproduction_summary
    
    def compare_with_original(self, reproduction_results: Dict[str, Any]):
        """Compare les résultats reproduits avec les originaux"""
        print("\n📊 COMPARAISON AVEC RÉSULTATS ORIGINAUX")
        print("=" * 50)
        
        # Résultats originaux de référence
        original_results = {
            'yolov12_sdpa': {'mAP50': 76.72, 'duration': 55.3},
            'yolov12_flash': {'mAP50': 76.53, 'duration': 67.3},
            'yolov13_sdpa': {'mAP50': 82.9, 'duration': 58.6},
            'yolov13_flash': {'mAP50': 82.3, 'duration': 65.7}
        }
        
        comparison = {}
        
        for exp_name in original_results.keys():
            if exp_name in reproduction_results['results']:
                repro_result = reproduction_results['results'][exp_name]
                original = original_results[exp_name]
                
                if repro_result['success']:
                    actual_map50 = repro_result['metrics']['mAP50'] * 100
                    duration = repro_result['duration_minutes']
                    
                    map50_diff = actual_map50 - original['mAP50']
                    duration_diff = duration - original['duration']
                    
                    comparison[exp_name] = {
                        'original_map50': original['mAP50'],
                        'reproduced_map50': actual_map50,
                        'map50_difference': map50_diff,
                        'original_duration': original['duration'],
                        'reproduced_duration': duration,
                        'duration_difference': duration_diff
                    }
                    
                    print(f"\n{exp_name.upper()}:")
                    print(f"  mAP50: {original['mAP50']:.2f}% → {actual_map50:.2f}% ({map50_diff:+.2f}%)")
                    print(f"  Durée: {original['duration']:.1f}min → {duration:.1f}min ({duration_diff:+.1f}min)")
        
        return comparison


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description='🔄 Reproduction Expériences YOLO')
    
    parser.add_argument('--config', type=str, choices=['yolov12_sdpa', 'yolov12_flash', 'yolov13_sdpa', 'yolov13_flash'],
                      help='Configuration spécifique à reproduire')
    parser.add_argument('--all', action='store_true', help='Reproduire toutes les expériences')
    parser.add_argument('--epochs', type=int, help='Nombre d\'époques (override)')
    parser.add_argument('--batch', type=int, help='Taille du batch (override)')
    parser.add_argument('--quick', action='store_true', help='Mode rapide (5 époques max)')
    parser.add_argument('--validate-env', action='store_true', help='Validation environnement seulement')
    
    args = parser.parse_args()
    
    print("🔄 REPRODUCTEUR EXPÉRIENCES YOLO")
    print("🎯 YOLOv12 vs YOLOv13 - SDPA vs Flash Attention")
    print("🇨🇩 Kennedy Kitoko - Agricultural AI Innovation")
    print("=" * 60)
    
    # Initialisation
    reproducer = ExperimentReproducer()
    
    # Validation environnement
    if not reproducer.validate_environment():
        print("❌ Environnement non prêt pour reproduction")
        return False
    
    if args.validate_env:
        print("✅ Validation environnement terminée")
        return True
    
    try:
        if args.config:
            # Reproduction expérience spécifique
            result = reproducer.reproduce_single_experiment(
                args.config, args.epochs, args.batch, args.quick
            )
            
            if result['success']:
                print(f"✅ Reproduction {args.config} réussie")
                return True
            else:
                print(f"❌ Reproduction {args.config} échouée")
                return False
                
        elif args.all:
            # Reproduction toutes expériences
            results = reproducer.reproduce_all_experiments(
                args.epochs, args.batch, args.quick
            )
            
            # Comparaison avec originaux
            reproducer.compare_with_original(results)
            
            success_rate = results['validated_reproductions'] / results['total_experiments']
            print(f"\n🎯 Taux de réussite reproduction: {success_rate:.1%}")
            
            return success_rate >= 0.75  # 75% minimum
            
        else:
            parser.print_help()
            return False
            
    except Exception as e:
        print(f"❌ Erreur reproduction: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    
    print(f"\n{'='*60}")
    print("🇨🇩 Kennedy Kitoko - Agricultural AI Innovation")
    print("🌍 Democratizing AI for Global Agriculture")
    print("🔄 Experiment Reproduction Framework")
    print("🔬 Ensuring Scientific Reproducibility")
    print(f"{'='*60}")
    
    sys.exit(0 if success else 1)