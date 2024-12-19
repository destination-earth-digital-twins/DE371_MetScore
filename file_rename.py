import os

# # Chemin du dossier contenant les fichiers à renommer
# dossier = '/project/scratch/p200177/DE_371/angeliquebonamy/results/dates/pack'
# #SCENARIOS:
# # Nouveau nom de base pour les fichiers
# # nouveau_nom = '_x_sample_68000'
# nouveau_nom = '__sample_'
# # Obtenir la liste des fichiers dans le dossier
# folders = sorted(os.listdir(dossier))
# for _,folder in enumerate(folders):
#     folder_path = os.path.join(dossier,folder)
#     print(folder_path)
# # Filtrer pour ne garder que les fichiers (pas les sous-dossiers)
# #fichiers = [f for f in fichiers if os.path.isfile(os.path.join(dossier, f))]
#     fichiers = sorted(os.listdir(folder_path))
#     # print(fichiers)
#    # print(fichiers)
#     j=0
# # # # Boucle sur chaque fichier et le renommer
#     for i, fichier in enumerate(fichiers, start=1):
#         # Obtenir l'extension du fichier
#         chemin_complet = os.path.join(folder_path,fichier)
#         # print(chemin_complet)
#         extension = os.path.splitext(fichier)[1]
        
# #         # if fichier.startswith('_'):
# #         #     nouveau_nom_complet = f"{nouveau_nom}_{i}{extension}"
# #         #     os.rename(os.path.join(folder_path,fichier), os.path.join(folder_path, nouveau_nom_complet))

#         if fichier.startswith('inv') and fichier.endswith('1500_mae.npy') or fichier.startswith('inv') and fichier.endswith('1000_.npy'):
#         # if fichier.startswith('_'):
#             j=j+1
#             # Créer le nouveau nom avec l'index pour le rendre unique
#             nouveau_nom_complet = f"{nouveau_nom}_{j}{extension}"
#         #  Renommer le fichier
#             os.rename(os.path.join(folder_path,fichier), os.path.join(folder_path, nouveau_nom_complet))
#         elif fichier.startswith('R'):
#             nouveau_nom_complet = f"{nouveau_nom}_{i}{extension}"
            
#             os.rename(os.path.join(folder_path,fichier), os.path.join(folder_path, nouveau_nom_complet))
#         elif fichier.startswith('inv'):
#             os.remove(chemin_complet)

#     print("Renommage terminé!")
    
#TOuT:
# Chemin du dossier contenant les fichiers à renommer
# dossier = '/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/mse/inversion/inversion'
# # Nouveau nom de base pour les fichiers
# nouveau_nom = '__sample_'
# # Obtenir la liste des fichiers dans le dossier
# folders = sorted(os.listdir(dossier))
# j=0
# i=0
# for _,file in enumerate(folders):
#     print(file)
#     file_path = os.path.join(dossier,file)
#     extension = os.path.splitext(file)[1]
#     print(file_path,extension)

#     if file.startswith('inv') and file.endswith('1500_mse.npy') or file.startswith('inv') and file.endswith('1500_.npy'):
#         # if fichier.startswith('_'):
#         j=j+1            # Créer le nouveau nom avec l'index pour le rendre unique
#         nouveau_nom_complet = f"{nouveau_nom}_{j}{extension}"
#         #  Renommer le fichier
#         os.rename(os.path.join(dossier,file), os.path.join(dossier, nouveau_nom_complet))
#     # if file.startswith('R'):
        
#     #     nouveau_nom_complet = f"{nouveau_nom}_{_}{extension}"
            
#     #     os.rename(os.path.join(dossier,file), os.path.join(dossier, nouveau_nom_complet))
#     elif file.startswith('inv'):
#         os.remove(file_path)

# print("Renommage terminé!")

import os
import re

# Fonction pour incrémenter un numéro spécifique dans un nom de fichier
def increment_file_number(folder_path):
    # Expression régulière pour correspondre aux fichiers commençant par 'Inv' et contenant un numéro
    regex_pattern = r"(InvertFsemble_\d{4}-\d{2}-\d{2}_)(\d+)(__\d+_\.npy)"

    for filename in os.listdir(folder_path):
        match = re.match(regex_pattern, filename)
        print(match)
        if match:
            # Extraire les parties du nom
            prefix, number, suffix = match.groups()

            # Incrémenter le numéro
            new_number = int(number) + 1

            # Construire le nouveau nom
            new_filename = f"{prefix}{new_number}{suffix}"

            # Renommer le fichier
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

# Spécifiez le chemin du dossier contenant les fichiers
folder = "chemin/vers/le/dossier"  # Remplacez par le chemin réel

# Dossier 1
folder = "/project/scratch/p200177/DE_371/angeliquebonamy/results/dates/test_scores/inv"  # Remplacez par le chemin du premier dossier
increment_file_number(folder)

# Dossier 2
# folder2 = "/project/scratch/p200177/DE_371/angeliquebonamy/results/dates/test_scores/pack"  # Remplacez par le chemin du deuxième dossier
# pattern2 = r"(Rsemble_\d{4}-\d{2}-\d{2}_)(\d+)(\.npy)"
# increment_file_number(folder2, pattern2, group_index_to_increment=1)
