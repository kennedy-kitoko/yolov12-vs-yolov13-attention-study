#!/usr/bin/env python3
"""
Script de vérification des dépendances pour YOLOv13
Vérifie toutes les librairies requises et leurs versions
"""

import sys
import subprocess
import importlib
from packaging import version
import warnings
warnings.filterwarnings("ignore")

# Liste des dépendances requises avec leurs versions
REQUIRED_PACKAGES = {
    'torch': '2.2.2',
    'torchvision': '0.17.2',
    'flash_attn': '2.7.3',  # Version spéciale avec CUDA
    'timm': '1.0.14',
    'albumentations': '2.0.4',
    'onnx': '1.14.0',
    'onnxruntime': '1.15.1',
    'pycocotools': '2.0.7',
    'PyYAML': '6.0.1',
    'scipy': '1.13.0',
    'onnxslim': '0.1.31',
    'onnxruntime-gpu': '1.18.0',
    'gradio': '4.44.1',
    'opencv-python': '4.9.0.80',
    'psutil': '5.9.8',
    'py-cpuinfo': '9.0.0',
    'huggingface-hub': '0.23.2',
    'safetensors': '0.4.3',
    'numpy': '1.26.4',
    'supervision': '0.22.0'
}

# Mapping des noms de packages pour l'import
IMPORT_MAPPING = {
    'opencv-python': 'cv2',
    'PyYAML': 'yaml',
    'py-cpuinfo': 'cpuinfo',
    'huggingface-hub': 'huggingface_hub',
    'onnxruntime-gpu': 'onnxruntime'
}

def check_python_version():
    """Vérifie la version de Python"""
    print("=" * 60)
    print("VÉRIFICATION DE L'ENVIRONNEMENT PYTHON")
    print("=" * 60)
    
    python_version = sys.version_info
    print(f"Version Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and python_version.minor == 11:
        print("✅ Version Python correcte (3.11)")
    else:
        print("❌ Version Python incorrecte. YOLOv13 nécessite Python 3.11")
    print()

def get_installed_version(package_name):
    """Récupère la version installée d'un package"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', package_name], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
        return None
    except:
        return None

def check_import(package_name):
    """Vérifie si un package peut être importé"""
    import_name = IMPORT_MAPPING.get(package_name, package_name.replace('-', '_'))
    
    try:
        if import_name == 'flash_attn':
            # Test spécial pour flash_attn
            import flash_attn
            return True, getattr(flash_attn, '__version__', 'unknown')
        else:
            module = importlib.import_module(import_name)
            module_version = getattr(module, '__version__', 'unknown')
            return True, module_version
    except ImportError as e:
        return False, str(e)

def check_cuda_availability():
    """Vérifie la disponibilité de CUDA"""
    print("=" * 60)
    print("VÉRIFICATION CUDA")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible")
            print(f"   Version CUDA: {torch.version.cuda}")
            print(f"   Nombre de GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("❌ CUDA non disponible")
    except ImportError:
        print("❌ PyTorch non installé, impossible de vérifier CUDA")
    print()

def check_flash_attention():
    """Vérifie spécifiquement Flash Attention"""
    print("VÉRIFICATION FLASH ATTENTION")
    print("-" * 40)
    
    try:
        import flash_attn
        print("✅ Flash Attention importé avec succès")
        
        # Teste une fonction de base
        try:
            from flash_attn import flash_attn_func
            print("✅ flash_attn_func disponible")
        except ImportError as e:
            print(f"⚠️  flash_attn_func non disponible: {e}")
            
    except ImportError as e:
        print(f"❌ Flash Attention non disponible: {e}")
        print("   Conseil: Installez avec la commande wheel spécifique fournie")
    print()

def main():
    """Fonction principale"""
    print("VÉRIFICATEUR DE DÉPENDANCES YOLOV13")
    print("=" * 60)
    print()
    
    # Vérification de la version Python
    check_python_version()
    
    # Vérification des packages
    print("=" * 60)
    print("VÉRIFICATION DES PACKAGES")
    print("=" * 60)
    
    all_ok = True
    results = []
    
    for package, required_version in REQUIRED_PACKAGES.items():
        print(f"Vérification de {package}...")
        
        # Vérifier la version installée
        installed_version = get_installed_version(package)
        
        # Vérifier l'import
        can_import, import_info = check_import(package)
        
        status = "✅"
        message = ""
        
        if installed_version is None:
            status = "❌"
            message = "Non installé"
            all_ok = False
        elif not can_import:
            status = "❌"
            message = f"Installé ({installed_version}) mais import échoué: {import_info}"
            all_ok = False
        else:
            # Comparer les versions (sauf pour flash_attn qui a une version spéciale)
            if package != 'flash_attn':
                try:
                    if version.parse(installed_version) < version.parse(required_version):
                        status = "⚠️"
                        message = f"Version {installed_version} < {required_version} (recommandée)"
                    elif version.parse(installed_version) > version.parse(required_version):
                        status = "⚠️"
                        message = f"Version {installed_version} > {required_version} (peut causer des incompatibilités)"
                    else:
                        message = f"Version {installed_version} ✓"
                except:
                    message = f"Version {installed_version} (vérification impossible)"
            else:
                message = f"Version {installed_version} (spéciale CUDA)"
        
        results.append((package, status, message))
        print(f"  {status} {package}: {message}")
    
    print()
    
    # Vérifications spéciales
    check_cuda_availability()
    check_flash_attention()
    
    # Résumé final
    print("=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    
    ok_count = sum(1 for _, status, _ in results if status == "✅")
    warning_count = sum(1 for _, status, _ in results if status == "⚠️")
    error_count = sum(1 for _, status, _ in results if status == "❌")
    
    print(f"✅ Packages OK: {ok_count}")
    print(f"⚠️  Packages avec avertissement: {warning_count}")
    print(f"❌ Packages manquants/défaillants: {error_count}")
    print()
    
    if error_count == 0:
        print("🎉 Tous les packages requis sont installés!")
        print("   Vous pouvez procéder à l'entraînement de YOLOv13.")
    else:
        print("❌ Des packages sont manquants ou défaillants.")
        print("   Installez les packages manquants avant de continuer.")
        
        print("\nCommandes d'installation suggérées:")
        print("pip install torch==2.2.2 torchvision==0.17.2")
        print("pip install timm==1.0.14 albumentations==2.0.4")
        print("pip install onnx==1.14.0 onnxruntime==1.15.1 onnxruntime-gpu==1.18.0")
        print("pip install pycocotools==2.0.7 PyYAML==6.0.1 scipy==1.13.0")
        print("pip install onnxslim==0.1.31 gradio==4.44.1")
        print("pip install opencv-python==4.9.0.80 psutil==5.9.8")
        print("pip install py-cpuinfo==9.0.0 huggingface-hub==0.23.2")
        print("pip install safetensors==0.4.3 numpy==1.26.4 supervision==0.22.0")
        print("pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")

if __name__ == "__main__":
    main()