import json
import re
from datasets import load_dataset

# Chemin de sortie
OUTPUT_PATH = r"C:\Users\clouduser\Desktop\CodeFuse-cgm\ground_truth.json"

def extract_files_from_patch(patch_text):
    """Extrait les noms de fichiers modifiés depuis le patch."""
    if not patch_text:
        return []
    
    files = set()
    # Regex standard pour les diffs git
    # Capture: diff --git a/chemin/fichier.py b/...
    matches = re.findall(r"diff --git a/(.*?) b/", patch_text)
    
    if not matches:
        # Fallback pour les diffs simples
        matches = re.findall(r"--- a/(.*)", patch_text)

    for f in matches:
        # Nettoyage et ajout
        filename = f.strip()
        if filename:
            files.add(filename)
            
    return list(files)

if __name__ == "__main__":
    print("🚀 Téléchargement de SWE-bench_Lite depuis Hugging Face...")
    
    # Chargement du split 'test' (c'est là où sont les bugs à résoudre)
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    print(f"✅ Dataset téléchargé : {len(dataset)} instances.")
    
    ground_truth = {}
    
    for row in dataset:
        instance_id = row['instance_id']
        patch_text = row['patch'] # La solution officielle
        
        modified_files = extract_files_from_patch(patch_text)
        
        if modified_files:
            ground_truth[instance_id] = modified_files
        else:
            print(f"⚠️ Pas de fichiers trouvés pour {instance_id}")

    # Sauvegarde
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=4)
        
    print(f"\n🎉 SUCCÈS TOTAL !")
    print(f"📂 Fichier Ground Truth créé ici : {OUTPUT_PATH}")
    
    # Vérification
    first_key = list(ground_truth.keys())[0]
    print(f"Exemple ({first_key}) : {ground_truth[first_key]}")