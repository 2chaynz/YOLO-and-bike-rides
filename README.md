# Pipeline de traitement des données

## Description

Ce projet analyse les données eye-tracking d'un utilisateur en vélo. Cela est la première étape dans notre Projet AZURE.

Le pipeline complet inclut :

1. **Tracking** : Détection et tracking de la chaussée dans les vidéos (YOLO)
2. **Projection** : Projection des points de regard sur les affiches (Homographie)

## Installation

Créez un environnement Python puis ;

Installez les dépendances :

```bash
pip install -r requirements.txt
```

## Utilisation du Pipeline

### Commande principale

```bash
# Une fois dans le bon dossier
# Pour le moment traite uniquement la vidéo que j'ai appelé vidéo1
# Modifer les paths dans le script pour appliquer le script a une autre vidéo.

python run_pipeline.py

```

### Configuration

Tous les paramètres sont centralisés dans `config.json` :

```json
{
    "_description": "Configuration pipeline pour video cycliste",
  
    "paths": {
        "video_input": "/Users/hobby/Desktop/Python/ML_projects/HACK sécu copie/Cyclistes/vid1/video1.mp4",
        "yolo_detection_weights": "yolov8n.pt",
        "output_dir": "/Users/hobby/Desktop/Python/ML_projects/HACK sécu copie/Cyclistes/vid1",
        "camera_params": "Users/hobby/Desktop/Python/ML_projects/HACK sécu copie/Cyclistes/vid1/scene_camera.json"
    },
  
    "settings": {
        "skip_step": 3,
        "start_frame": 0,
        "end_frame": null,
        "detection_confidence": 0.3,
        "tracking_iou": 0.5
    },
  
    "files_associated": {
        "notes": "Si données d'eye-tracker associées à cette vidéo a mettre ici",
        "gaze": "gaze.csv",
        "world_timestamps": "world_timestamps.csv"
    }
}

```

## Structure du Projet

```

├──Cyclistes/
│   ├── vid1/    # Données brutes des sujets
│   │   ├── *.mp4              # Vidéo de la scène
│   │   ├── *_output.mp4              # Vidéo avec gaze et boxes (génerée)
│   │   ├── gaze.csv           # Données de regard
│   │   ├── scene_camera.json  # Paramètres de caméra
│   │   ├── gaze_result.csv  # (a générer) Projections de regard
│   │   ├── ***.csv + ***.json  # autres données issues des lunettes
│   │   └── detection_results.csv  # (a générer) Détections YOLO
│   │ 
│   └── vid2/ ...
│
└── Code video 1
     ├──src/
     │	│
     │	├── detectionModel/
     │	│   └── DetectionModel.py      # Wrapper YOLO
     │	└── utils/
     │   	└── DataLoader.py          # Chargement des données
     │
     ├── run_pipeline.py          # Script principal du pipeline
     ├── config.json              # Configuration centralisée
     ├── .gitignore  
     ├── README.md 
     ├── yolov8.pt                # Modèle de regression entrainé sur COCO récupéré
     └── requirements.txt         # Dépendances Python

```

## Étapes du Pipeline A ADAPTER CAR ICI POUR POSTERS

### Étape 1 : Tracking (YOLO)

Génère un fichier `detection_results.csv` pour chaque sujet contenant les bounding boxes des affiches détectées à chaque frame.

```
frame	track_id	class_id	class_name	conf	x1	y1	x2
0	1	2	car	0.9015472531318665	0	407	413
0	2	2	car	0.8738754391670227	276	357	580
0	3	2	car	0.8097224235534668	528	345	650
0	4	2	car	0.4880787134170532	639	328	714
...
```

### Étape 2 : Projection des regards

Génère un fichier `gaze_projections_subjectX.csv` contenant les coordonnées du regard projeté sur chaque objet.

```
frame_idx	timestamp_ns	gaze_x_px	gaze_y_px	object_id	object_class
0	1763649011136000000	927	174	-1	none
5	1763649011386000000	927	174	-1	none
10	1763649011636000000	927	174	-1	none
15	1763649011886000000	927	174	-1	none
...
```

## Références

- [Automatic Detection and Rectification of Paper Receipts on Smartphones](https://arxiv.org/pdf/2303.05763)
