"""
Pipeline principal de traitement des données de regard.

Ce script permet de traiter les données eye-tracking pour :
1. Générer les fichiers de détection/tracking (detection_results.csv)
2. Projeter les points de regard (gaze_projections.csv)

Usage (une fois dans le bon dossier):
    python run_pipeline.py                 
"""
import json
import os
import cv2
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.detectionModel.DetectionModel import DetectionModel
from src.utils.DataLoader import load_camera_params 

def run_full_pipeline(config):
    paths = config["paths"]
    video_path = paths["video_input"]
    output_dir = paths["output_dir"]
    camera_json = paths.get("camera_params")
    
    # Paths
    gaze_csv = "/Users/hobby/Desktop/Python/ML_projects/HACK sécu copie/Cyclistes/vid1/gaze.csv"
    timestamps_csv = "/Users/hobby/Desktop/Python/ML_projects/HACK sécu copie/Cyclistes/vid1/world_timestamps.csv"
    
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, "resultat_gaze_optimise.mp4")
    
    # Fichiers CSV de sortie
    detection_csv = os.path.join(output_dir, "detection_results.csv")
    gaze_projection_csv = os.path.join(output_dir, "gaze_projections.csv")

    # 1. Chargement Gaze & Timestamps
    print("ℹ Chargement des fichiers de données...")
    df_gaze = pd.read_csv(gaze_csv)
    df_ts = pd.read_csv(timestamps_csv)

    # 2. Modèle & Calibration
    K, D = None, None
    if camera_json and os.path.exists(camera_json):
        K, D = load_camera_params(camera_json)
    model = DetectionModel(paths["yolo_detection_weights"])
    
    # 3. Ouverture Vidéo
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- REGLAGE OPTIMISATION ---
    skip_step = 1 
    new_fps = original_fps / skip_step # La vidéo restera à vitesse réelle

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video, fourcc, new_fps, (width, height))

    # 4. Ouverture des fichiers CSV pour écriture
    detection_file = open(detection_csv, 'w', newline='')
    detection_writer = csv.writer(detection_file)
    detection_writer.writerow(['frame', 'track_id', 'class_id', 'class_name', 'conf', 'x1', 'y1', 'x2', 'y2'])
    
    gaze_file = open(gaze_projection_csv, 'w', newline='')
    gaze_writer = csv.writer(gaze_file)
    gaze_writer.writerow(['frame_idx', 'timestamp_ns', 'gaze_x_px', 'gaze_y_px', 'object_id', 'object_class'])

    print(f" Tracking + Gaze (1 frame sur {skip_step}). Sortie : {output_video}")

    for frame_idx in tqdm(range(0, total_frames, skip_step), desc="Traitement"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: 
            break
        
        # A. Undistort
        if K is not None and D is not None:
            frame = cv2.undistort(frame, K, D)

        # B. Tracking YOLO
        results = model.model.track(
            source=frame, 
            persist=True, 
            verbose=False, 
            conf=0.3
        )

        detected_objects = []  # Pour stocker les objets détectés de cette frame
        
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                # Coordonnées
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Classe et confiance
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.model.names[class_id]
                
                # Track ID (peut être None si tracking échoue)
                track_id = int(box.id[0]) if box.id is not None else -1
                
                # Enregistrer dans le CSV de détection
                detection_writer.writerow([
                    frame_idx, 
                    track_id, 
                    class_id, 
                    label, 
                    conf, 
                    x1, y1, x2, y2
                ])
                
                # Stocker pour vérifier intersection avec gaze
                detected_objects.append({
                    'track_id': track_id,
                    'class': label,
                    'bbox': (x1, y1, x2, y2)
                })
                
                # Couleur selon la classe (style YOLO classique)
                colors = {
                    'car': (0, 0, 255),
                    'truck': (203, 192, 255),
                    'bus': (255, 0, 255),
                    'person': (238, 130, 238),
                    'bicycle': (255, 255, 0),
                    'motorcycle': (255, 0, 0),
                    'traffic light': (0, 255, 0),
                    'stop sign': (255, 128, 0),
                }
                color = colors.get(label, (0, 0, 255))  # Rouge par défaut
                
                # Dessiner la boite
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label avec confiance
                text = f"{label} {conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(frame, 
                            (x1, y1 - text_height - baseline - 5), 
                            (x1 + text_width, y1), 
                            color, -1)
                cv2.putText(frame, text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # C. Projection Gaze (Synchronisation par Timestamps)
        try:
            if frame_idx < len(df_ts):
                current_ts = df_ts.iloc[frame_idx]['timestamp [ns]']
                
                # Recherche du point de regard le plus proche
                idx_gaze = (df_gaze['timestamp [ns]'] - current_ts).abs().idxmin()
                gaze_row = df_gaze.loc[idx_gaze]
                
                # Conversion depuis les pixels
                g_x = int(float(gaze_row['gaze x [px]']))
                g_y = int(float(gaze_row['gaze y [px]']))

                # Vérifier si le gaze intersecte un objet détecté
                intersected_object_id = -1
                intersected_object_class = 'none'
                
                for obj in detected_objects:
                    x1, y1, x2, y2 = obj['bbox']
                    if x1 <= g_x <= x2 and y1 <= g_y <= y2:
                        intersected_object_id = obj['track_id']
                        intersected_object_class = obj['class']
                        break
                
                # Enregistrer dans le CSV de gaze
                gaze_writer.writerow([
                    frame_idx,
                    current_ts,
                    g_x,
                    g_y,
                    intersected_object_id,
                    intersected_object_class
                ])

                # Dessin du point de regard
                cv2.circle(frame, (g_x, g_y), 10, (0, 0, 255), -1)
                cv2.circle(frame, (g_x, g_y), 11, (255, 255, 255), 1)
                
        except Exception as e:
            print(f" Warning frame {frame_idx}: {e}")

        # Vérification que frame est valide avant d'écrire
        if frame is not None and frame.size > 0:
            out_video.write(frame)

    # Fermeture des fichiers
    detection_file.close()
    gaze_file.close()
    cap.release()
    out_video.release()
    
    print(f"\n Terminé ! Fichiers générés :")
    print(f"   - Vidéo : {output_video}")
    print(f"   - Détections : {detection_csv}")
    print(f"   - Projections gaze : {gaze_projection_csv}")

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config_data = json.load(f)
    run_full_pipeline(config_data)
