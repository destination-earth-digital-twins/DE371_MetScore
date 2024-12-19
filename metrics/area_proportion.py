import numpy as np
import os
import matplotlib.pyplot as plt


def area_greater_than(
    data, variable, threshs=[0, 1, 3, 5, 10, 15, 20, 25, 30]#, 40, 50]
):
    """
    Extract from each grid all the values greater than threshold and compute their area proportion

    Args:
        variable (int): index of the channels corresponding to the variable
        data (array): array of the loaded data, shape N x C x H x W
        thresholds (list):me    list of thresholds
    Returns:
        np.array[float]: store every value greater than the threshold
    """
    mean_proportion = np.zeros((len(threshs),))
    print('JE SUIS LA SAPE DANS AREA PROPORTION')
    for idx_threshold, threshold in enumerate(threshs):
        # mask = np.exp((data[:, variable]+1)*5.78319931/2)-1 > threshold
        # extracted = (np.exp((data[:, variable]+1)*5.78319931/2)-1)[mask]
        mask = data[:, variable] > threshold
        extracted = data[:, variable][mask]
        mean_proportion[idx_threshold] += len(extracted)

    mean_proportion /= data.shape[-2] * data.shape[-1] * data.shape[0]

    return mean_proportion


# for _,scenarios in enumerate(os.listdir('/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/BON/mse/pack/')):
#    # print(scenarios)
#     thresholds=[0, 1, 3, 5, 10, 15, 20, 25, 30, 40, 50]
#     path_inv = os.path.join('/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/BON/VGG_trained/sol2/scores',scenarios)
#     path_pack = os.path.join('/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/BON/VGG_trained/sol2/pack',scenarios)

#     area_propor_inv = np.zeros(len(thresholds))
#     area_propor_pack = np.zeros(len(thresholds))

#     for _,file in enumerate(os.listdir(path_inv)):
#         file_path = os.path.join(path_inv,file)
#         if file_path.endswith('1500_.npy'):
#             data = np.load(file_path)
#             area_propor_inv=area_greater_than(data,0,thresholds)+area_propor_inv
#     for _,file in enumerate(os.listdir(path_pack )):
#         file_path = os.path.join(path_pack ,file)
#         data = np.load(file_path)
#         area_propor_pack+=area_greater_than(data,0,thresholds)
    
#     print(scenarios,area_propor_inv)
#     plt.figure()
#     plt.plot(thresholds,area_propor_inv, label='Inv')
#     plt.plot(thresholds,area_propor_pack, label='Arome')

#     # Tracer la deuxième série de donnéesplt.plot([0, 1, 3, 5, 10, 15, 20, 25, 30, 40, 50], area_propor_inv, label='Inv')

#     # Appliquer une échelle logarithmique à l'axe des ordonnées
#     plt.yscale('log')

#         # Ajouter un titre, une légende et des labels
#     plt.title(scenarios)
#     plt.xlabel('thresh')
#     plt.ylabel('pixels ')
#     plt.legend()
#     plt.savefig(f'/project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/BON/VGG_trained/sol2/scores/{scenarios}/{scenarios}_VGG_train_sol2.png')
# # /project/scratch/p200177/DE_371/angeliquebonamy/results/scenarios/BON/mse/scores/Alpes-Mar_Golfe-G/scores

#     # Afficher le graphique
    
# # area_propor_pack = 0
# # for _,file in enumerate(os.listdir(path_pack )):
# #     file_path = os.path.join(path_pack ,file)
# #     data = np.load(file_path)
# #     area_propor_pack+=area_greater_than(data,0,thresholds)

# # # Tracer la première série de données
# # plt.plot(thresholds,area_propor_pack, label='Arome')

# # # Tracer la deuxième série de données
# # plt.plot([0, 1, 3, 5, 10, 15, 20, 25, 30, 40, 50], area_propor_inv, label='Inv')

# # # Appliquer une échelle logarithmique à l'axe des ordonnées
# # plt.yscale('log')

# # # Ajouter un titre, une légende et des labels
# # plt.title('Centre Médit')
# # plt.xlabel('thresh')
# # plt.ylabel('pixels ')
# # plt.legend()
# # plt.savefig('Centre Médit_mse.png')

# # # Afficher le graphique
# # plt.show()
