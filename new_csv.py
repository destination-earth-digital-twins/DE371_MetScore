import pandas as pd

# Charger les fichiers CSV
# csv1 = pd.read_csv("IS_boostrap_no_duplicate_rr_cumul_correct_valid.csv")  # Premier CSV avec les données initiales
# csv2 = pd.read_csv("labels.csv")  # Deuxième CSV avec toutes les combinaisons

# # Identifier les dates uniques dans le premier CSV
# dates_in_csv1 = csv1["Date"].unique()

# # Filtrer le deuxième CSV pour ne garder que les dates présentes dans le premier CSV
# csv2_filtered = csv2[csv2["Date"].isin(dates_in_csv1)]

# # Trouver les LeadTimes manquants
# merged = pd.merge(csv2_filtered, csv1, on=["Date", "Leadtime", "Member", "Gigafile", "Localindex"], how="left", suffixes=("", "_y"))

# # Les lignes manquantes dans le premier CSV auront des valeurs NaN dans les colonnes du premier CSV
# missing_rows = merged[merged["Name_y"].isna()]

# # Créer les nouvelles entrées avec des valeurs par défaut pour les colonnes manquantes
# missing_rows = missing_rows[["Name", "Date", "Leadtime", "Member", "Gigafile", "Localindex"]]
# missing_rows["Importance"] = None  # Par défaut, aucune importance

# # Ajouter les lignes manquantes au premier CSV
# updated_csv1 = pd.concat([csv1, missing_rows], ignore_index=True)

# # Sauvegarder le CSV mis à jour
# updated_csv1.to_csv("updated_file1.csv", index=False)

# print("Mise à jour terminée. Les LeadTimes manquants ont été ajoutés.")


# Cette partie sert à transformer le train csv, utiliser grâce à l'IS pour prendre les ensembles entiers (membres + leadtime)

csv1 = pd.read_csv("/project/home/p200177/DE_371/datasets/dataset_Meteo_France_rr_u_v_t2m/data/IS_rr_debug_1_1.0_0_0_0_0_0_256_large_lt/IS_boostrap_no_duplicate_rr_cumul_correct_train.csv")  # Premier CSV avec les données initiales
csv2 = pd.read_csv("/project/home/p200177/DE_371/datasets/dataset_Meteo_France_rr_u_v_t2m/data/data_for_importance_sampling/labels.csv")  # Deuxième CSV avec toutes les combinaisons

# Charger les fichiers CSV
input_file = csv1  # Remplacez par le chemin de votre fichier
output_file = "output.csv"
all_members_file = csv2  # Chemin du fichier contenant tous les membres
completed_output_file = "completed_output.csv"



import pandas as pd
import numpy
import matplotlib.pyplot as plt


# Charger les fichiers CSV
df_complet = csv2
df_sample = pd.read_csv('output.csv')
print(df_complet['Date'].describe())
# Créer l'ensemble complet des combinaisons possibles (date, leadtime, Membre)
combinaisons_completes = df_complet[['Date', 'Leadtime', 'Member']].drop_duplicates()

# Trouver les combinaisons manquantes dans le dataset avec importance sampling
combinaisons_sample = df_sample[['Date', 'Leadtime', 'Member']].drop_duplicates()
manquantes = combinaisons_completes.merge(
    combinaisons_sample, 
    on=['Date', 'Leadtime', 'Member'], 
    how='left', 
    indicator=True
).query('_merge == "left_only"').drop('_merge', axis=1)

# Ajouter les colonnes manquantes pour compléter les données
# Assigner des valeurs par défaut ou NaN pour les colonnes non spécifiées
colonnes_manquantes = set(df_sample.columns) - set(manquantes.columns)
for col in colonnes_manquantes:
    manquantes[col] = None  # ou une valeur par défaut

# Combiner les données manquantes avec le dataset sample
df_complet_sample = pd.concat([df_sample, manquantes], ignore_index=True)

# Sauvegarder le nouveau dataset
df_complet_sample.to_csv('dataset_sample_complet.csv', index=False)

print("Le dataset a été complété avec les combinaisons manquantes.")

# # Fonction pour compléter les données
# completed_data = []
# for key in output_data['Key'].unique():
#     # Filtrer les données par clé
#     output_group = output_data[output_data['Key'] == key]
#     all_members_group = all_members_data[all_members_data['Key'] == key]

    # Vérifier le nombre de membres présents
#     if output_group['Member'].nunique() < 16:
#         missing_members = set(all_members_group['Member']) - set(output_group['Member'])
#         for member in missing_members:
#             missing_row = all_members_group[all_members_group['Member'] == member]
#             print('MEMEBERS',all_members_group[all_members_group['Member'] == member],'MISSING ROW', missing_row.iloc[0])
#             completed_data.append(missing_row.iloc[0])

#     # Ajouter les données existantes
#     completed_data.append(output_group)
#     print(completed_data)
# # Combiner toutes les données en un DataFrame
# completed_data = pd.concat(completed_data, ignore_index=True)

# # Sauvegarder le résultat
# completed_data.to_csv(completed_output_file, index=False)

print(f"Fichier CSV complété avec 16 membres par Date et LeadTime créé : {completed_output_file}")

