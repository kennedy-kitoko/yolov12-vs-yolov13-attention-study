#!/usr/bin/env python3
"""
🚀 LANCEUR EXPÉRIENCES COMPARATIVES YOLO COMPLET
YOLOv12 vs YOLOv13 avec SDPA vs Flash Attention
Développé par Kennedy Kitoko 🇨🇩
"""

import os
import sys
import json
import time
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

class ComprehensiveYOLOExperimentLauncher:
    """
    Lanceur d'expériences comparatives pour YOLOv12 vs YOLOv13
    avec mécanismes d'attention SDPA vs Flash Attention
    """
    
    def __init__(self, dataset_path: str = None):
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = f"yolo_comparative_experiments_{self.session_id}"
        
        # Configuration dataset optimisée pour Kennedy
        self.dataset_paths = self._find_dataset_paths(dataset_path)
        
        # Configuration expériences optimisée pour le setup
        self.optimized_config = {
            'cpu_cores': 12,
            'ram_gb': 39,
            'recommended_epochs': 100,
            'recommended_batch': 16,
            'workers': 8,  # 2/3 des cores pour DataLoader
            'mixed_precision': True,  # AMP avec beaucoup de RAM
            'cache_images': True,  # Cache avec 39GB RAM
            'multi_scale_training': True
        }
        
        self.results = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': datetime.now().isoformat(),
                'researcher': 'Kennedy Kitoko 🇨🇩',
                'experiment_type': 'YOLOv12 vs YOLOv13 - SDPA vs Flash Attention Comprehensive Comparison',
                'dataset_used': None
            },
            'validation_results': {},
            'experiments': {
                'yolov12_sdpa': {},
                'yolov12_flash': {},
                'yolov13_sdpa': {},
                'yolov13_flash': {}
            },
            'comparative_analysis': {},
            'session_summary': {}
        }
        
        # Création structure session
        os.makedirs(self.base_dir, exist_ok=True)
        
        print(f"🚀 Lanceur d'expériences YOLO comparatives initialisé")
        print(f"📁 Session: {self.session_id}")
        print(f"📂 Dossier: {self.base_dir}")
    
    def _find_dataset_paths(self, custom_path: str = None) -> List[str]:
        """Recherche intelligente du dataset"""
        paths_to_check = []
        
        if custom_path:
            paths_to_check.append(custom_path)
        
        # Ajout path Kennedy dans la recherche
        common_paths = [
            '/mnt/c/Users/kitok/Desktop/SmartFarm_OS/IA/YOLO12_FLASH_attn/Weeds-3',  # Path Kennedy
            'Weeds-3',
            './Weeds-3',
            '../Weeds-3',
            '../../Weeds-3',
            '/content/Weeds-3',  # Google Colab
            '/kaggle/input/weeds-3',  # Kaggle
            '/mnt/c/Users/kitok/Desktop/SmartFarm_OS/IA/datasets/Weeds-3',  # Path alternatif Kennedy
        ]
        
        paths_to_check.extend(common_paths)
        
        valid_paths = []
        for path in paths_to_check:
            if os.path.exists(os.path.join(path, 'train', 'images')):
                valid_paths.append(os.path.abspath(path))
        
        return valid_paths
    
    def comprehensive_validation(self) -> bool:
        """Validation complète de l'environnement"""
        print("\n🔍 VALIDATION COMPLÈTE DE L'ENVIRONNEMENT...")
        
        validation = {
            'python_ok': False,
            'torch_ok': False,
            'cuda_ok': False,
            'ultralytics_ok': False,
            'flash_attn_ok': False,
            'pillow_ok': False,
            'dataset_ok': False,
            'yolov12_weights_ok': False,
            'yolov13_weights_ok': False,
            'sdpa_backends_ok': False
        }
        
        # Python
        if sys.version_info >= (3, 8):
            validation['python_ok'] = True
            print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        else:
            print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} (requis: ≥3.8)")
        
        # PyTorch + CUDA
        try:
            import torch
            validation['torch_ok'] = True
            print(f"✅ PyTorch {torch.__version__}")

            if torch.cuda.is_available():
                validation['cuda_ok'] = True
                device_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"✅ CUDA - {device_name} ({memory_gb:.1f} GB)")
            else:
                print("❌ CUDA non disponible")
        except ImportError:
            print("❌ PyTorch non disponible")
        except Exception as e:
            print(f"❌ Erreur lors de la vérification PyTorch/CUDA : {e}")
        
        # Test SDPA backends avec gestion d'erreurs
        try:
            # Tentative import avec gestion d'erreurs pour PyTorch 2.2.2
            try:
                from torch.nn.attention import SDPBackend
                backends = []
                if hasattr(torch.backends.cuda, 'flash_sdp_enabled') and torch.backends.cuda.flash_sdp_enabled():
                    backends.append("Flash")
                if hasattr(torch.backends.cuda, 'mem_efficient_sdp_enabled') and torch.backends.cuda.mem_efficient_sdp_enabled():
                    backends.append("Efficient")
                if hasattr(torch.backends.cuda, 'math_sdp_enabled') and torch.backends.cuda.math_sdp_enabled():
                    backends.append("Math")

                if backends:
                    validation['sdpa_backends_ok'] = True
                    print(f"✅ SDPA Backends: {', '.join(backends)}")
                else:
                    print("⚠️ SDPA Backends: Aucun backend détecté")

            except ImportError:
                # Fallback pour PyTorch 2.2.2
                print("⚠️ torch.nn.attention non disponible - PyTorch 2.2.2 compatibility")
                validation['sdpa_backends_ok'] = True  # Assume SDPA is available

        except Exception as e:
            print(f"⚠️ SDPA Backends: Erreur {e}")
        
        # Ultralytics
        try:
            from ultralytics import YOLO
            validation['ultralytics_ok'] = True
            print("✅ Ultralytics YOLO")
            
            # Test téléchargement modèles
            try:
                # YOLOv12
                yolo12_model = YOLO('yolo12n.pt')
                validation['yolov12_weights_ok'] = True
                print("✅ YOLOv12-N weights disponibles")
            except Exception as e:
                print(f"⚠️ YOLOv12-N weights: {e}")
            
            try:
                # YOLOv13 (hypothétique - adapter selon disponibilité)
                yolo13_model = YOLO('yolov13n.pt')
                validation['yolov13_weights_ok'] = True
                print("✅ YOLOv13-N weights disponibles")
            except Exception as e:
                print(f"⚠️ YOLOv13-N weights: {e}")
                
        except ImportError:
            print("❌ Ultralytics non disponible")
        
        # Flash Attention
        try:
            import flash_attn
            validation['flash_attn_ok'] = True
            version = getattr(flash_attn, '__version__', 'unknown')
            print(f"✅ Flash Attention {version}")
        except ImportError:
            print("⚠️ Flash Attention non disponible (non bloquant)")
        
        # Pillow
        try:
            from PIL import Image
            validation['pillow_ok'] = True
            print("✅ Pillow (PIL)")
        except ImportError:
            print("⚠️ Pillow non accessible")
        
        # Dataset
        if self.dataset_paths:
            validation['dataset_ok'] = True
            dataset_path = self.dataset_paths[0]
            self.results['session_info']['dataset_used'] = dataset_path
            print(f"✅ Dataset: {dataset_path}")
            
            # Vérification structure
            train_images = len(list(Path(dataset_path, 'train', 'images').glob('*')))
            val_images = len(list(Path(dataset_path, 'val', 'images').glob('*')))
            print(f"   📊 Images: {train_images} train, {val_images} val")
        else:
            print("❌ Dataset non trouvé")
        
        # Résumé validation
        critical_checks = ['python_ok', 'torch_ok', 'cuda_ok', 'ultralytics_ok', 'dataset_ok']
        critical_passed = sum(validation[check] for check in critical_checks)
        total_passed = sum(validation.values())
        
        ready = critical_passed >= 5 and validation['yolov12_weights_ok']
        
        print(f"\n📊 Validation: {total_passed}/{len(validation)} checks")
        print(f"🎯 Critique: {critical_passed}/{len(critical_checks)} essentiels")
        print(f"🏁 Prêt pour expériences: {'Oui' if ready else 'Non'}")
        
        self.results['validation_results'] = {
            'checks': validation,
            'ready_for_experiments': ready,
            'critical_passed': critical_passed,
            'total_passed': total_passed
        }
        
        return ready
    
    def get_memory_usage(self) -> float:
        """Utilisation mémoire actuelle en MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        """Utilisation mémoire GPU (used, total) en GB"""
        try:
            import torch
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated(0) / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                return used, total
        except:
            pass
        return 0.0, 0.0
    
    def run_yolo_experiment(self, 
                          model_version: str, 
                          attention_type: str,
                          epochs: int = 100,  # Optimisé pour setup Kennedy
                          batch_size: int = 16) -> Dict[str, Any]:  # Batch plus grand
        """
        Expérience YOLO générique
        
        Args:
            model_version: 'yolov12' ou 'yolov13'
            attention_type: 'sdpa' ou 'flash'
            epochs: Nombre d'époques
            batch_size: Taille du batch
        """
        experiment_name = f"{model_version}_{attention_type}"
        print(f"\n🔬 EXPÉRIENCE {experiment_name.upper()}...")
        
        try:
            # Import PyTorch en premier
            import torch
            from ultralytics import YOLO
            
            # Configuration optimisée pour setup Kennedy
            if model_version == "yolov12":
                model_file = "yolo12n.pt"
            elif model_version == "yolov13":
                model_file = "yolov13n.pt"
            else:
                model_file = f"{model_version}n.pt"
            data_file = os.path.join(self.dataset_paths[0], 'data.yaml')
            
            config = {
                'model': model_file,
                'data': data_file,
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': 640,
                'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
                'amp': True,  # Mixed precision activé avec beaucoup de RAM
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'lrf': 0.01,  # Learning rate final
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'patience': 50,  # Plus de patience avec plus d'époques
                'project': self.base_dir,
                'name': experiment_name,
                'save_period': 25,  # Sauvegarde tous les 25 époques
                'workers': self.optimized_config['workers'],  # 8 workers
                'cache': 'ram',  # Cache en RAM avec 39GB
                'rect': True,  # Rectangular training
                'cos_lr': True,  # Cosine LR scheduler
                'close_mosaic': 10,  # Disable mosaic last 10 epochs
                'resume': False,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'verbose': True,
                # Optimisations spécifiques setup Kennedy
                'multi_scale': True,  # Multi-scale training
                'copy_paste': 0.1,  # Copy-paste augmentation
                'mixup': 0.1,  # Mixup augmentation
                'hsv_h': 0.015,  # HSV-Hue augmentation
                'hsv_s': 0.7,   # HSV-Saturation
                'hsv_v': 0.4,   # HSV-Value
                'degrees': 0.0,  # Rotation degrees
                'translate': 0.1,  # Translation
                'scale': 0.5,   # Scale
                'shear': 0.0,   # Shear
                'perspective': 0.0,  # Perspective
                'flipud': 0.0,  # Flip up-down
                'fliplr': 0.5,  # Flip left-right
                'mosaic': 1.0,  # Mosaic augmentation
            }
            
            # Import et monitoring
            start_time = time.time()
            start_memory = self.get_memory_usage()
            start_gpu_memory, total_gpu_memory = self.get_gpu_memory_usage()
            
            # Configuration attention selon PyTorch 2.2.2
            if attention_type == 'sdpa':
                print("🎯 Configuration SDPA...")
                os.environ['TORCH_CUDNN_BENCHMARK'] = '1'
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                
                # Configuration SDPA pour PyTorch 2.2.2
                try:
                    # Essai avec la nouvelle API
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                    torch.backends.cuda.enable_math_sdp(True)
                    print("✅ SDPA configuré via torch.backends")
                except AttributeError:
                    # Fallback pour versions antérieures
                    print("⚠️ Using fallback SDPA configuration")
                
            elif attention_type == 'flash':
                print("🔥 Configuration Flash Attention...")
                try:
                    import flash_attn
                    print(f"✅ Flash Attention {getattr(flash_attn, '__version__', 'unknown')} détecté")
                    
                    os.environ['FLASH_ATTENTION_FORCE_USE'] = '1'
                    os.environ['FLASH_ATTENTION_SKIP_CUDA_BUILD'] = '0'
                    
                    # Configuration Flash Attention
                    try:
                        torch.backends.cuda.enable_flash_sdp(True)
                        torch.backends.cuda.enable_mem_efficient_sdp(False)
                        torch.backends.cuda.enable_math_sdp(False)
                        print("✅ Flash Attention configuré via torch.backends")
                    except AttributeError:
                        print("⚠️ Using fallback Flash Attention configuration")
                        
                except ImportError:
                    print("❌ Flash Attention non disponible, basculement vers SDPA")
                    return self.run_yolo_experiment(model_version, 'sdpa', epochs, batch_size)
            
            print("📋 Configuration:")
            for k, v in config.items():
                print(f"   {k}: {v}")
            
            # Chargement modèle
            print(f"📦 Chargement {model_file}...")
            model = YOLO(config['model'])
            
            # YOLOv13 spécifique - Configuration HyperACE si disponible
            if model_version == 'yolov13' and attention_type == 'flash':
                try:
                    # Hypothétique: activation HyperACE avec Flash Attention
                    print("🧠 Tentative activation HyperACE avec Flash Attention...")
                    # model.model.enable_hyperace(attention='flash')  # API hypothétique
                except Exception as e:
                    print(f"⚠️ HyperACE non configuré: {e}")
            
            # Entraînement
            print("🚀 Démarrage entraînement...")
            results = model.train(**{k: v for k, v in config.items() if k not in ['model']})
            
            # Métriques finales
            end_time = time.time()
            end_memory = self.get_memory_usage()
            end_gpu_memory, _ = self.get_gpu_memory_usage()
            duration = end_time - start_time
            
            # Extraction métriques détaillées
            metrics = {}
            if hasattr(results, 'results_dict'):
                metrics = {
                    'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
                    'mAP50_95': results.results_dict.get('metrics/mAP50-95(B)', 0),
                    'precision': results.results_dict.get('metrics/precision(B)', 0),
                    'recall': results.results_dict.get('metrics/recall(B)', 0),
                    'fitness': results.results_dict.get('fitness', 0)
                }
            
            # Résultats expérience
            experiment_result = {
                'success': True,
                'model_version': model_version,
                'attention_type': attention_type,
                'duration_seconds': duration,
                'duration_minutes': duration / 60,
                'config': config,
                'metrics': metrics,
                'performance': {
                    'memory_start_mb': start_memory,
                    'memory_end_mb': end_memory,
                    'memory_used_mb': end_memory - start_memory,
                    'gpu_memory_start_gb': start_gpu_memory,
                    'gpu_memory_end_gb': end_gpu_memory,
                    'gpu_memory_used_gb': end_gpu_memory - start_gpu_memory,
                    'total_gpu_memory_gb': total_gpu_memory
                },
                'results_path': str(results.save_dir) if hasattr(results, 'save_dir') else None,
                'best_weights': None,
                'timestamp': datetime.now().isoformat()
            }
            
            # Recherche meilleurs poids
            if experiment_result['results_path']:
                weights_dir = Path(experiment_result['results_path']) / 'weights'
                if weights_dir.exists():
                    best_pt = weights_dir / 'best.pt'
                    if best_pt.exists():
                        experiment_result['best_weights'] = str(best_pt)
            
            print(f"✅ {experiment_name.upper()} terminé en {duration/60:.1f} minutes")
            print(f"📊 Mémoire utilisée: {end_memory - start_memory:.1f} MB")
            print(f"🎮 GPU mémoire utilisée: {end_gpu_memory - start_gpu_memory:.2f} GB")
            if metrics:
                print(f"📈 mAP50: {metrics.get('mAP50', 0):.3f}")
                print(f"📈 mAP50-95: {metrics.get('mAP50_95', 0):.3f}")
            
            return experiment_result
            
        except Exception as e:
            print(f"❌ Erreur {experiment_name}: {e}")
            print(f"🔍 Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'model_version': model_version,
                'attention_type': attention_type,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'duration_seconds': time.time() - start_time if 'start_time' in locals() else 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_all_experiments(self, epochs: int = 100, batch_size: int = 16) -> bool:
        """Lance toutes les expériences comparatives"""
        print(f"\n{'='*60}")
        print("🚀 LANCEMENT EXPÉRIENCES COMPARATIVES COMPLÈTES")
        print(f"{'='*60}")
        
        experiments_config = [
        #    ('yolov12', 'sdpa'),
        #    ('yolov12', 'flash'),
        #    ('yolov13', 'sdpa'),
            ('yolov13', 'flash')
        ]
        
        successful_experiments = 0
        
        for i, (model_version, attention_type) in enumerate(experiments_config, 1):
            print(f"\n{'='*20}")
            print(f"EXPÉRIENCE {i}/4: {model_version.upper()} + {attention_type.upper()}")
            print(f"{'='*20}")
            
            result = self.run_yolo_experiment(model_version, attention_type, epochs, batch_size)
            experiment_key = f"{model_version}_{attention_type}"
            self.results['experiments'][experiment_key] = result
            
            if result['success']:
                successful_experiments += 1
                print(f"✅ Expérience {experiment_key} réussie")
            else:
                print(f"❌ Expérience {experiment_key} échouée")
            
            # Pause entre expériences pour libérer mémoire
            if i < len(experiments_config):
                print("⏸️ Pause nettoyage mémoire...")
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    time.sleep(5)
                except:
                    pass
        
        print(f"\n📊 Expériences terminées: {successful_experiments}/{len(experiments_config)}")
        return successful_experiments > 0
    
    def comprehensive_analysis(self) -> Dict[str, Any]:
        """Analyse comparative complète des résultats"""
        print(f"\n{'='*60}")
        print("📊 ANALYSE COMPARATIVE COMPLÈTE")
        print(f"{'='*60}")
        
        try:
            experiments = self.results['experiments']
            successful_experiments = {k: v for k, v in experiments.items() if v.get('success', False)}
            
            if not successful_experiments:
                print("❌ Aucune expérience réussie pour analyse")
                return {}
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'successful_experiments': list(successful_experiments.keys()),
                'failed_experiments': [k for k, v in experiments.items() if not v.get('success', False)],
                'model_comparison': {},
                'attention_comparison': {},
                'performance_ranking': {},
                'recommendations': []
            }
            
            # Comparaison par modèle
            yolov12_results = {k: v for k, v in successful_experiments.items() if 'yolov12' in k}
            yolov13_results = {k: v for k, v in successful_experiments.items() if 'yolov13' in k}
            
            if yolov12_results and yolov13_results:
                print("\n🔍 COMPARAISON MODÈLES YOLOv12 vs YOLOv13")
                
                # Moyennes YOLOv12
                yolov12_metrics = self._calculate_average_metrics(yolov12_results)
                # Moyennes YOLOv13
                yolov13_metrics = self._calculate_average_metrics(yolov13_results)
                
                analysis['model_comparison'] = {
                    'yolov12_average': yolov12_metrics,
                    'yolov13_average': yolov13_metrics,
                    'winner': 'yolov13' if yolov13_metrics['mAP50'] > yolov12_metrics['mAP50'] else 'yolov12',
                    'mAP_difference': abs(yolov13_metrics['mAP50'] - yolov12_metrics['mAP50']),
                    'speed_difference_minutes': abs(yolov13_metrics['duration_minutes'] - yolov12_metrics['duration_minutes'])
                }
                
                print(f"YOLOv12 mAP50 moyen: {yolov12_metrics['mAP50']:.3f}")
                print(f"YOLOv13 mAP50 moyen: {yolov13_metrics['mAP50']:.3f}")
                print(f"Gagnant modèle: {analysis['model_comparison']['winner'].upper()}")
            
            # Comparaison par attention
            sdpa_results = {k: v for k, v in successful_experiments.items() if 'sdpa' in k}
            flash_results = {k: v for k, v in successful_experiments.items() if 'flash' in k}
            
            if sdpa_results and flash_results:
                print("\n🔍 COMPARAISON ATTENTION SDPA vs Flash")
                
                sdpa_metrics = self._calculate_average_metrics(sdpa_results)
                flash_metrics = self._calculate_average_metrics(flash_results)
                
                analysis['attention_comparison'] = {
                    'sdpa_average': sdpa_metrics,
                    'flash_average': flash_metrics,
                    'winner': 'flash' if flash_metrics['mAP50'] > sdpa_metrics['mAP50'] else 'sdpa',
                    'mAP_difference': abs(flash_metrics['mAP50'] - sdpa_metrics['mAP50']),
                    'speed_difference_minutes': abs(flash_metrics['duration_minutes'] - sdpa_metrics['duration_minutes'])
                }
                
                print(f"SDPA mAP50 moyen: {sdpa_metrics['mAP50']:.3f}")
                print(f"Flash mAP50 moyen: {flash_metrics['mAP50']:.3f}")
                print(f"Gagnant attention: {analysis['attention_comparison']['winner'].upper()}")
            
            # Ranking performance
            ranking = []
            for exp_name, exp_data in successful_experiments.items():
                metrics = exp_data.get('metrics', {})
                performance = exp_data.get('performance', {})
                
                ranking.append({
                    'experiment': exp_name,
                    'mAP50': metrics.get('mAP50', 0),
                    'mAP50_95': metrics.get('mAP50_95', 0),
                    'duration_minutes': exp_data.get('duration_minutes', 0),
                    'memory_used_mb': performance.get('memory_used_mb', 0),
                    'gpu_memory_used_gb': performance.get('gpu_memory_used_gb', 0)
                })
            
            # Tri par mAP50 décroissant
            ranking.sort(key=lambda x: x['mAP50'], reverse=True)
            analysis['performance_ranking'] = ranking
            
            print(f"\n🏆 CLASSEMENT PERFORMANCE:")
            for i, exp in enumerate(ranking, 1):
                print(f"{i}. {exp['experiment']}: mAP50={exp['mAP50']:.3f}, "
                      f"Durée={exp['duration_minutes']:.1f}min")
            
            # Recommandations
            if ranking:
                best_experiment = ranking[0]
                analysis['recommendations'] = [
                    f"Meilleure configuration: {best_experiment['experiment']}",
                    f"mAP50 optimale: {best_experiment['mAP50']:.3f}",
                    f"Durée d'entraînement: {best_experiment['duration_minutes']:.1f} minutes"
                ]
                
                if len(ranking) > 1:
                    speed_ranking = sorted(ranking, key=lambda x: x['duration_minutes'])
                    fastest = speed_ranking[0]
                    if fastest['experiment'] != best_experiment['experiment']:
                        analysis['recommendations'].append(
                            f"Configuration la plus rapide: {fastest['experiment']} "
                            f"({fastest['duration_minutes']:.1f}min)"
                        )
            
            # Sauvegarde analyse
            analysis_file = f"{self.base_dir}/comprehensive_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=4, default=str)
            
            print(f"\n💾 Analyse sauvegardée: {analysis_file}")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Erreur analyse: {e}")
            return {'error': str(e)}
    
    def _calculate_average_metrics(self, experiment_results: Dict[str, Any]) -> Dict[str, float]:
        """Calcule les métriques moyennes pour un ensemble d'expériences"""
        if not experiment_results:
            return {}
        
        metrics_sums = {
            'mAP50': 0,
            'mAP50_95': 0,
            'duration_minutes': 0,
            'memory_used_mb': 0,
            'gpu_memory_used_gb': 0
        }
        
        count = len(experiment_results)
        
        for exp_data in experiment_results.values():
            metrics = exp_data.get('metrics', {})
            performance = exp_data.get('performance', {})
            
            metrics_sums['mAP50'] += metrics.get('mAP50', 0)
            metrics_sums['mAP50_95'] += metrics.get('mAP50_95', 0)
            metrics_sums['duration_minutes'] += exp_data.get('duration_minutes', 0)
            metrics_sums['memory_used_mb'] += performance.get('memory_used_mb', 0)
            metrics_sums['gpu_memory_used_gb'] += performance.get('gpu_memory_used_gb', 0)
        
        return {k: v / count for k, v in metrics_sums.items()}
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Génère un rapport final complet"""
        print(f"\n{'='*60}")
        print("📋 GÉNÉRATION RAPPORT FINAL COMPLET")
        print(f"{'='*60}")
        
        try:
            session_end = datetime.now()
            session_start = datetime.fromisoformat(self.results['session_info']['start_time'])
            total_duration = (session_end - session_start).total_seconds()
            
            # Résumé session
            experiments = self.results['experiments']
            successful_count = sum(1 for exp in experiments.values() if exp.get('success', False))
            
            summary = {
                'session_duration_minutes': total_duration / 60,
                'total_experiments': len(experiments),
                'successful_experiments': successful_count,
                'failed_experiments': len(experiments) - successful_count,
                'validation_success': self.results.get('validation_results', {}).get('ready_for_experiments', False),
                'dataset_used': self.results['session_info']['dataset_used'],
                'research_success': successful_count > 0
            }
            
            # Sauvegarde résultats complets
            self.results['session_summary'] = summary
            results_file = f"{self.base_dir}/complete_session_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=4, default=str)
            
            # Rapport lisible
            report_file = f"{self.base_dir}/comprehensive_report.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# 🚀 RAPPORT EXPÉRIENCES COMPARATIVES YOLO\n\n")
                print("## Spécifications Setup\n")
                f.write(f"- **CPU:** AMD Ryzen 9 7945HX (12 cores)\n")
                f.write(f"- **RAM:** 39 GB disponible\n")
                f.write(f"- **Stockage:** 899 GB libre\n")
                f.write(f"- **OS:** Linux (WSL2)\n")
                f.write(f"- **Environnement:** flash-attention conda env\n\n")
                f.write(f"- **Session ID:** {self.session_id}\n")
                f.write(f"- **Durée totale:** {summary['session_duration_minutes']:.1f} minutes\n")
                f.write(f"- **Date:** {session_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- **Dataset:** {summary['dataset_used']}\n")
                f.write(f"- **Chercheur:** Kennedy Kitoko 🇨🇩\n\n")
                
                f.write("## Résultats Expériences\n")
                f.write(f"- **Total:** {summary['total_experiments']}\n")
                f.write(f"- **Réussies:** {summary['successful_experiments']}\n")
                f.write(f"- **Échouées:** {summary['failed_experiments']}\n\n")
                
                # Détails par expérience
                f.write("### Détails par Expérience\n\n")
                for exp_name, exp_data in experiments.items():
                    status = "✅ Réussie" if exp_data.get('success', False) else "❌ Échouée"
                    f.write(f"#### {exp_name.upper()} - {status}\n")
                    
                    if exp_data.get('success', False):
                        metrics = exp_data.get('metrics', {})
                        f.write(f"- **Durée:** {exp_data.get('duration_minutes', 0):.1f} minutes\n")
                        f.write(f"- **mAP50:** {metrics.get('mAP50', 0):.3f}\n")
                        f.write(f"- **mAP50-95:** {metrics.get('mAP50_95', 0):.3f}\n")
                        f.write(f"- **Précision:** {metrics.get('precision', 0):.3f}\n")
                        f.write(f"- **Rappel:** {metrics.get('recall', 0):.3f}\n")
                        
                        performance = exp_data.get('performance', {})
                        f.write(f"- **Mémoire utilisée:** {performance.get('memory_used_mb', 0):.1f} MB\n")
                        f.write(f"- **Mémoire GPU:** {performance.get('gpu_memory_used_gb', 0):.2f} GB\n")
                        
                        if exp_data.get('results_path'):
                            f.write(f"- **Résultats:** {exp_data['results_path']}\n")
                    else:
                        f.write(f"- **Erreur:** {exp_data.get('error', 'Inconnue')}\n")
                    
                    f.write("\n")
                
                # Analyse comparative
                analysis = self.results.get('comparative_analysis', {})
                if analysis:
                    f.write("## Analyse Comparative\n\n")
                    
                    if 'model_comparison' in analysis:
                        comp = analysis['model_comparison']
                        f.write("### YOLOv12 vs YOLOv13\n")
                        f.write(f"- **Gagnant:** {comp.get('winner', 'N/A').upper()}\n")
                        f.write(f"- **Différence mAP50:** {comp.get('mAP_difference', 0):.3f}\n")
                        f.write(f"- **Différence vitesse:** {comp.get('speed_difference_minutes', 0):.1f} min\n\n")
                    
                    if 'attention_comparison' in analysis:
                        comp = analysis['attention_comparison']
                        f.write("### SDPA vs Flash Attention\n")
                        f.write(f"- **Gagnant:** {comp.get('winner', 'N/A').upper()}\n")
                        f.write(f"- **Différence mAP50:** {comp.get('mAP_difference', 0):.3f}\n")
                        f.write(f"- **Différence vitesse:** {comp.get('speed_difference_minutes', 0):.1f} min\n\n")
                    
                    if 'performance_ranking' in analysis:
                        f.write("### Classement Performance\n")
                        for i, exp in enumerate(analysis['performance_ranking'], 1):
                            f.write(f"{i}. **{exp['experiment']}:** mAP50={exp['mAP50']:.3f}, "
                                  f"Durée={exp['duration_minutes']:.1f}min\n")
                        f.write("\n")
                    
                    if 'recommendations' in analysis:
                        f.write("### Recommandations\n")
                        for rec in analysis['recommendations']:
                            f.write(f"- {rec}\n")
                        f.write("\n")
                
                f.write("## Innovations Techniques\n\n")
                f.write("### YOLOv13 - Innovations\n")
                f.write("- **HyperACE:** Mécanisme d'attention basé sur les hypergraphes\n")
                f.write("- **FullPAD:** Paradigme de distribution complète des caractéristiques\n")
                f.write("- **DS-Blocks:** Blocs de convolution séparable en profondeur\n\n")
                
                f.write("### Flash Attention vs SDPA\n")
                f.write("- **SDPA:** Attention native PyTorch optimisée\n")
                f.write("- **Flash Attention:** Optimisation mémoire avec réordonnancement\n")
                f.write("- **Complexité:** Linéaire vs quadratique pour les longues séquences\n\n")
                
                f.write("---\n")
                f.write("**Développé par Kennedy Kitoko 🇨🇩**\n")
                f.write("*Démocratisation de l'IA pour l'Agriculture Mondiale*\n")
            
            print(f"💾 Rapport complet: {report_file}")
            print(f"💾 Données JSON: {results_file}")
            
            return summary
            
        except Exception as e:
            print(f"❌ Erreur rapport final: {e}")
            return {'error': str(e)}
    
    def run_complete_workflow(self, 
                            epochs: int = 100,  # Optimisé setup Kennedy
                            batch_size: int = 16,  # Batch plus grand
                            skip_yolov13: bool = False) -> bool:
        """
        Workflow complet d'expériences comparatives
        
        Args:
            epochs: Nombre d'époques par expérience
            batch_size: Taille du batch
            skip_yolov13: Ignorer YOLOv13 si non disponible
        """
        print("🚀 WORKFLOW COMPLET D'EXPÉRIENCES COMPARATIVES YOLO")
        print("🇨🇩 Kennedy Kitoko - Agricultural AI Innovation")
        print("=" * 70)
        
        print(f"\n📋 Configuration:")
        print(f"   Époques: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Skip YOLOv13: {skip_yolov13}")
        
        try:
            # Phase 1: Validation
            print(f"\n{'='*20}")
            print("PHASE 1: VALIDATION ENVIRONNEMENT")
            print(f"{'='*20}")
            
            if not self.comprehensive_validation():
                print("❌ Validation échouée - Arrêt workflow")
                return False
            
            # Vérification YOLOv13
            if not skip_yolov13 and not self.results['validation_results']['checks']['yolov13_weights_ok']:
                print("⚠️ YOLOv13 non disponible - Passage en mode YOLOv12 seulement")
                skip_yolov13 = True
            
            # Phase 2: Expériences
            print(f"\n{'='*20}")
            print("PHASE 2: EXPÉRIENCES COMPARATIVES")
            print(f"{'='*20}")
            
            if skip_yolov13:
                # Mode YOLOv12 seulement
                print("🎯 Mode YOLOv12 uniquement")
                experiments_success = True
                
                # YOLOv12 + SDPA
                result_sdpa = self.run_yolo_experiment('yolov12', 'sdpa', epochs, batch_size)
                self.results['experiments']['yolov12_sdpa'] = result_sdpa
                
                # YOLOv12 + Flash
                result_flash = self.run_yolo_experiment('yolov12', 'flash', epochs, batch_size)
                self.results['experiments']['yolov12_flash'] = result_flash
                
                experiments_success = result_sdpa['success'] or result_flash['success']
            else:
                # Mode complet
                experiments_success = self.run_all_experiments(epochs, batch_size)
            
            # Phase 3: Analyse
            print(f"\n{'='*20}")
            print("PHASE 3: ANALYSE COMPARATIVE")
            print(f"{'='*20}")
            
            analysis = self.comprehensive_analysis()
            self.results['comparative_analysis'] = analysis
            
            # Phase 4: Rapport final
            print(f"\n{'='*20}")
            print("PHASE 4: RAPPORT FINAL")
            print(f"{'='*20}")
            
            summary = self.generate_comprehensive_report()
            
            # Résumé final
            print(f"\n{'='*30}")
            print("🎉 WORKFLOW COMPARATIF TERMINÉ")
            print(f"{'='*30}")
            
            if summary.get('research_success', False):
                print("🏆 SUCCÈS! Expériences comparatives terminées")
                print("📊 Données scientifiques collectées et analysées")
                
                # Affichage résumé rapide
                successful = summary['successful_experiments']
                total = summary['total_experiments']
                print(f"✅ Expériences réussies: {successful}/{total}")
                
                if 'comparative_analysis' in self.results and self.results['comparative_analysis']:
                    analysis = self.results['comparative_analysis']
                    
                    if 'performance_ranking' in analysis and analysis['performance_ranking']:
                        best = analysis['performance_ranking'][0]
                        print(f"🥇 Meilleure configuration: {best['experiment']}")
                        print(f"📈 mAP50 optimal: {best['mAP50']:.3f}")
                
            else:
                print("⚠️ Workflow partiel - Certaines expériences ont échoué")
                print("💾 Données partielles sauvegardées")
            
            print(f"📁 Tous les résultats dans: {self.base_dir}")
            
            return summary.get('research_success', False)
            
        except Exception as e:
            print(f"❌ Erreur workflow: {e}")
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False


def main():
    """Point d'entrée principal"""
    print("🚀 LANCEUR EXPÉRIENCES COMPARATIVES YOLO")
    print("🎯 YOLOv12 vs YOLOv13 - SDPA vs Flash Attention")
    print("🇨🇩 Kennedy Kitoko - AI Democratization & Agricultural Innovation")
    print("=" * 70)
    
    print("\n🧠 Contexte Technique:")
    print("📋 YOLOv12: Area Attention (A2) + R-ELAN + FlashAttention")
    print("📋 YOLOv13: HyperACE (Hypergraph) + FullPAD + DS-Blocks")
    print("📋 SDPA: Scaled Dot-Product Attention natif PyTorch")
    print("📋 Flash Attention: Optimisation mémoire IO-aware")
    
    print("\n🔬 Expériences Planifiées:")
    print("1. YOLOv12 + SDPA")
    print("2. YOLOv12 + Flash Attention")
    print("3. YOLOv13 + SDPA")
    print("4. YOLOv13 + Flash Attention")
    
    # Configuration optimisée pour setup Kennedy
    # Setup détecté: AMD Ryzen 9 7945HX, 39GB RAM, 899GB libre
    default_epochs = 100  # Plus d'époques pour meilleure convergence
    default_batch = 16    # Batch plus grand avec 39GB RAM
    dataset_path = "/mnt/c/Users/kitok/Desktop/SmartFarm_OS/IA/YOLO12_FLASH_attn/Weeds-3"
    
    print(f"\n⚙️ Configuration optimisée pour votre setup:")
    print(f"   💻 CPU: AMD Ryzen 9 7945HX (12 cores)")
    print(f"   🧠 RAM: 39GB disponible")
    print(f"   💾 Espace: 899GB libre")
    print(f"   📊 Époques: {default_epochs} (convergence optimale)")
    print(f"   🔢 Batch size: {default_batch} (optimisé RAM)")
    print(f"   📁 Dataset: Weeds-3 pré-configuré")
    print(f"   ⏱️ Durée estimée: 4-8 heures (entraînement complet)")
    print(f"   💽 Espace requis: ~15 GB (tous modèles)")
    
    # Options interactives
    try:
        print(f"\n🔧 Configuration (Entrée = défaut):")
        
        # Époques
        epochs_input = input(f"Époques [{default_epochs}]: ").strip()
        epochs = int(epochs_input) if epochs_input else default_epochs
        
        # Batch size
        batch_input = input(f"Batch size [{default_batch}]: ").strip()
        batch_size = int(batch_input) if batch_input else default_batch
        
        # Dataset personnalisé
        dataset_input = input("Chemin dataset [auto]: ").strip()
        dataset_path = dataset_input if dataset_input else None
        
        # Mode rapide optimisé pour tests
        quick_mode = input("Mode rapide (50 époques au lieu de 100) ? [n]: ").strip().lower()
        if quick_mode in ['y', 'yes', 'oui']:
            epochs = 50
            print(f"⚡ Mode rapide activé - Époques: {epochs}")
        
        # Mode ultra-rapide pour développement
        dev_mode = input("Mode développeur (20 époques, tests rapides) ? [n]: ").strip().lower()
        if dev_mode in ['y', 'yes', 'oui']:
            epochs = 20
            batch_size = min(batch_size, 8)  # Batch plus petit pour tests
            print(f"🔧 Mode développeur activé - Époques: {epochs}, Batch: {batch_size}")
        
        # GPU detection et optimisation batch
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"\n🎮 GPU détecté: {gpu_name} ({gpu_memory:.1f} GB)")
                
                # Ajustement batch selon GPU
                if gpu_memory >= 24:  # RTX 4090, A100, etc.
                    suggested_batch = min(batch_size * 2, 32)
                    increase_batch = input(f"GPU puissant détecté. Augmenter batch à {suggested_batch}? [n]: ").strip().lower()
                    if increase_batch in ['y', 'yes', 'oui']:
                        batch_size = suggested_batch
                        print(f"🚀 Batch augmenté à {batch_size}")
                elif gpu_memory < 8:  # GPU plus modeste
                    suggested_batch = max(batch_size // 2, 4)
                    print(f"⚠️ GPU modeste détecté. Réduction batch recommandée: {suggested_batch}")
                    reduce_batch = input(f"Réduire batch à {suggested_batch}? [y]: ").strip().lower()
                    if reduce_batch not in ['n', 'no', 'non']:
                        batch_size = suggested_batch
                        print(f"📉 Batch réduit à {batch_size}")
            else:
                print("⚠️ Pas de GPU détecté - Mode CPU (très lent)")
                epochs = min(epochs, 10)
                batch_size = min(batch_size, 4)
                print(f"🐌 Mode CPU: Époques={epochs}, Batch={batch_size}")
                
        except ImportError:
            pass
        
        # Confirmation finale
        print(f"\n📋 Configuration finale:")
        print(f"   Époques: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Dataset: {dataset_path or 'Auto-détection'}")
        
        confirm = input("\n🚀 Lancer les expériences comparatives? [Y/n]: ").strip().lower()
        if confirm in ['n', 'no', 'non']:
            print("❌ Expériences annulées")
            return False
            
    except KeyboardInterrupt:
        print("\n❌ Expériences annulées par l'utilisateur")
        return False
    except ValueError as e:
        print(f"❌ Erreur configuration: {e}")
        return False
    
    # Lancement expériences
    launcher = ComprehensiveYOLOExperimentLauncher(dataset_path)
    
    try:
        success = launcher.run_complete_workflow(
            epochs=epochs,
            batch_size=batch_size,
            skip_yolov13=False  # Tentera YOLOv13, passera en mode YOLOv12 si non disponible
        )
        
        if success:
            print("\n🏆 EXPÉRIENCES COMPARATIVES RÉUSSIES!")
            print("📊 Analyse complète YOLOv12 vs YOLOv13")
            print("🔬 Comparaison SDPA vs Flash Attention")
            print("📈 Métriques de performance collectées")
        else:
            print("\n📋 EXPÉRIENCES PARTIELLES")
            print("💾 Certaines données collectées")
            print("🔍 Vérifier les logs pour plus de détails")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Erreur expériences: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    
    print(f"\n{'='*70}")
    print("🇨🇩 Kennedy Kitoko - Agricultural AI Innovation")
    print("🌍 Democratizing AI for Global Agriculture")
    print("🧠 HyperACE: Hypergraph-Enhanced Adaptive Visual Perception")
    print("⚡ Flash Attention: Memory-Efficient IO-Aware Computation")
    print("📚 SDPA: Scaled Dot-Product Attention Excellence")
    print(f"{'='*70}")
    
    sys.exit(0 if success else 1)