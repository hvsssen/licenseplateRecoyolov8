from ultralytics import YOLO
import pytesseract
import cv2
import pandas as pd
import mysql.connector
import re
from rapidfuzz import fuzz  # For fuzzy string matching
# Connexion à MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",  
    password="root", 
    database="plaque_db" 
)
cursor = db.cursor()

# Nom de la table basé sur le nom de la vidéo
video_name = "try.mp4"
table_name = "platesfor" + video_name.split('.')[0] +'video'

# Création de la table unique avec une colonne 'confidence'
cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        number_plate VARCHAR(20) NOT NULL UNIQUE,
        confidence FLOAT NOT NULL
    )
""")
db.commit()

# Chargement du modèle YOLO
model = YOLO('best.pt')

# Chargement de la vidéo
cap = cv2.VideoCapture(video_name)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Seuils et structures de stockage
CONFIDENCE_THRESHOLD = 0.7  # Seuil de confiance minimum
DETECTION_THRESHOLD = 5  # Nombre minimum de détections avant insertion
plate_detections = {}  
print(f"Initial plate_detections: {plate_detections}")# Stocke les plaques détectées avec leurs niveaux de confiance
current_plate = None  # Tracks the plate currently being "paused" on
pause =None
SIMILARITY_THRESHOLD = 85  # Similarity percentage to consider plates the same (adjustable)

def find_similar_plate(new_text, plate_dict):
    new_normalized = normalize_plate(new_text)
    for normalized_key, (original, _) in plate_dict.items():
        similarity = fuzz.ratio(new_normalized, normalized_key)
        if similarity >= SIMILARITY_THRESHOLD:
            return normalized_key, original
    return None, None
def normalize_plate(text):
    return re.sub(r'[\s-]', '', text).upper()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.predict(frame, stream=True)
    
    for r in results:
        boxes = r.boxes.xyxy
        confidences = r.boxes.conf
        annotated_frame = r.plot()
        
        if len(boxes) > 0:
            boxes_df = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2'])
            boxes_df['confidence'] = confidences.cpu().numpy()

            for _, row in boxes_df.iterrows():
                x1, y1, x2, y2, conf = map(float, [row['x1'], row['y1'], row['x2'], row['y2'], row['confidence']])

                if conf < CONFIDENCE_THRESHOLD:
                    continue
                
                cropped = frame[int(y1):int(y2), int(x1):int(x2)]
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                thresh = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
               
                 # Set width=600, height=300
                cv2.imshow("Processed Plate", thresh)
                crop_img_text = pytesseract.image_to_string(thresh, lang='eng').strip()

                if not crop_img_text:
                    continue
                print(f"Texte détecté: {crop_img_text} (Confiance: {conf:.2f})")

                # Vérification et nettoyage du texte
                if len(crop_img_text) > 5 and not re.search(r'[!*\'\+\(\)%/?~|,«—:\[\]§.°¥a-z]', crop_img_text):
                    crop_img_text = crop_img_text.strip(" -")
                    normalized_text = normalize_plate(crop_img_text)  # Version normalisée
                    # If we’re paused on a plate, check if this is different
                    if current_plate is not None:
                        similarity = fuzz.ratio(normalized_text, normalize_plate(current_plate))
                        if similarity >= SIMILARITY_THRESHOLD:
                            print(f"Paused: Still detecting similar plate {current_plate} (detected: {crop_img_text}, similarity: {similarity:.1f}%), skipping frame.")
                            continue # Skip this frame if it’s the same plate
                        else:
                            print(f"New plate {crop_img_text} detected, resuming processing.")
                            current_plate = None  # Reset to resume normal processing
                    
                    # Stocker toutes les confiances pour cette plaque
                    if crop_img_text not in plate_detections:
                        plate_detections[crop_img_text] = []
                    
                    plate_detections[crop_img_text].append(conf)
                    
                    # Vérifier si cette plaque a été détectée plus de 5 fois
                    if len(plate_detections[crop_img_text]) > DETECTION_THRESHOLD:
                        print("Plaque " + crop_img_text + " " + str(plate_detections[crop_img_text]))
                        best_conf = max(plate_detections[crop_img_text])
                        print(f" ***************** license with best best  conf: {crop_img_text} (Confiance: {best_conf:.2f})")

                        # Vérifier si la plaque est déjà enregistrée en base
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE number_plate = %s", (crop_img_text,))
                        already_exists = cursor.fetchone()[0]

                        if already_exists == 0:
                            current_plate= crop_img_text
                            print(f"✅ Insertion de la plaque {crop_img_text} avec confiance {best_conf:.2f}")
                            cursor.execute(f"INSERT INTO {table_name} (number_plate, confidence) VALUES (%s, %s)", 
                                           (crop_img_text, best_conf))
                            db.commit()
                        
                        # Set this as the current plate and pause further processing
                        del plate_detections[crop_img_text]  # Clear detections for this plate
    annotated_frame = cv2.resize(annotated_frame, (700, 700))
    cv2.imshow("YOLO Detection", annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
db.close()
model.close()