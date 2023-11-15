import cv2
import numpy as np
import os
import pandas as pd
import random

def apply_sharpening_filter(image):
    sharpening_filter = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]], dtype=np.float32)
    sharpened_image = cv2.filter2D(image, -1, sharpening_filter)
    return sharpened_image

def calculate_segmented_volume(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    distance_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
    _, sure_fg = cv2.threshold(distance_transform, 0.7 * distance_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(thresh, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [0, 255, 0]
    segmented_volume = np.sum(np.all(image == [0, 255, 0], axis=-1))
    #cv2.imshow("Segmented Image", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return segmented_volume
def classify_image(image_path, class_average_volumes):
    # Charger l'image
    if not os.path.exists(image_path):
        print(f"Error: Image file does not exist at {image_path}")
        return None

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    # Calculer le volume de la partie segmentée de l'image
    segmented_volume = calculate_segmented_volume(image)
    print(segmented_volume)
    if segmented_volume is None:
        print(f"Error: Unable to calculate segmented volume for image at {image_path}")
        return None

    # Trouver la classe la plus proche en fonction du volume
    closest_class = min(class_average_volumes, key=lambda x: abs(class_average_volumes[x] - segmented_volume))

    return closest_class


def main():
    # Chemin vers le répertoire contenant les images de la dataset
    dataset_path = './Alzheimer_s Dataset/train'

    # Liste des noms de classes
    classes = ['ModerateDemented', 'VeryMildDemented', 'NonDemented']

    # Dictionnaire pour stocker les volumes moyens de chaque classe
    class_average_volumes = {}

    # Nombre d'images à traiter par classe
    sample_size = 51

    # Boucle sur chaque classe
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        image_files = os.listdir(class_path)

        # Sélectionnez un échantillon aléatoire de 51 images
        random.shuffle(image_files)
        image_files = image_files[:sample_size]

        # Initialiser la liste des volumes de la classe
        class_volumes = []

        # Boucle sur chaque image de la classe
        for image_file in image_files:
            # Construire le chemin complet de l'image
            image_path = os.path.join(class_path, image_file)

            # Charger l'image IRM
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            # Calculer le volume de la partie segmentée de l'image
            segmented_volume = calculate_segmented_volume(image)

            # Ajouter le volume à la liste des volumes de la classe
            class_volumes.append(segmented_volume)

        # Calculer la moyenne des volumes de la classe
        class_average_volume = np.mean(class_volumes)

        # Ajouter le volume moyen de la classe au dictionnaire
        class_average_volumes[class_name] = class_average_volume

    # Enregistrer les volumes moyens dans un fichier CSV
    df = pd.DataFrame(class_average_volumes.items(), columns=['Classe', 'Volume moyen'])
    df.to_csv(os.path.join(os.path.dirname(__file__), 'volumes_moyens.csv'), index=False)

    # Charger les volumes moyens à partir du fichier CSV
    volumes_df = pd.read_csv('./volumes_moyens.csv')
    class_average_volumes = {row['Classe']: row['Volume moyen'] for index, row in volumes_df.iterrows()}

    # Chemin de l'image à classer
    image_path_to_classify = './Alzheimer_s Dataset/train/NonDemented/nonDem112.jpg'

    # Classer l'image en utilisant la fonction de classification
    predicted_class = classify_image(image_path_to_classify, class_average_volumes)

    # Afficher le résultat en fonction de la classe prédite
    if predicted_class is not None:
        if predicted_class in ["ModerateDemented", "VeryMildDemented"]:
            print('Demented')
        else:
            print('Non Demented')
if __name__ == "__main__":
    main()
