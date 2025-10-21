from flask import Flask, render_template, request, send_from_directory, jsonify
import cv2
from ultralytics import YOLO
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from model import ConvNeXtWithECA, TripletNet
import atexit
import shutil
from collections import defaultdict

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1
model_yolo = YOLO('detecting.pt')
COUNT = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_backbone = ConvNeXtWithECA()
model_embed = TripletNet(embedding_backbone)
model_embed.load_state_dict(torch.load("triplet_convnext_eca.pth", map_location=device))
model_embed.to(device)
model_embed.eval()

gallery_data = torch.load("gallery_embeddings.pt", map_location="cpu")
gallery_vectors = gallery_data["vectors"]  
gallery_labels = gallery_data["labels"]    

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@app.route('/')
def man():
    return render_template('home.html')

def extract_embedding(image_pil):
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model_embed.embeddingnet(image_tensor)
    return embedding.cpu()  

@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img_file = request.files['image']
    if not img_file:
        return jsonify({"error": "No file provided"}), 400

    img_path = f'static/{COUNT}.jpg'
    img_file.save(img_path)

    img_cv2 = cv2.imread(img_path)
    results = model_yolo(img_cv2, save=True)[0] 
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        return jsonify({"error": "No tire track detected"}), 400
    x1, y1, x2, y2 = boxes.xyxy[0].int().tolist()
    crop = img_cv2[y1:y2, x1:x2]

    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    query_vec = extract_embedding(crop_pil)

    sims = cosine_similarity(query_vec, gallery_vectors).flatten()  # (N,)
    # # sim>0.7
    # label_scores = defaultdict(float)
    # for i, score in enumerate(sims):
    #     label = gallery_labels[i]
    #     if score > 0.7 and score > label_scores[label]:
    #         label_scores[label] = score
    # matched_labels = sorted(label_scores.keys(), key=lambda x: -label_scores[x])
    # if not matched_labels:
    #     matched_labels = ["No matching class found"]

    # ✅ Lấy top 5 similarity cao nhất
    sorted_indices = np.argsort(sims)[::-1]
    top_k_unique = []
    seen_labels = set()

    for idx in sorted_indices:
        label = gallery_labels[idx]
        sim_score = float(sims[idx])
        if label not in seen_labels and sim_score >= 0.75:
            top_k_unique.append((label, sim_score))
            seen_labels.add(label)
        if len(top_k_unique) == 5:
            break

    if not top_k_unique:
        top_k_unique = [("No matching class found", 0.0)]
    COUNT += 1
    return render_template(
        'detection.html',
        matched_results=top_k_unique
)


@app.route('/load_img')
def load_img():
    return send_from_directory('runs/detect/predict', "image0.jpg")

@app.route('/img')
def img():
    return send_from_directory('static', f"{COUNT-1}.jpg")

def cleanup_runs():
    if os.path.exists('runs'):
        try:
            shutil.rmtree('runs')
            print("✅ Deleted runs/ folder.")
        except Exception as e:
            print(f"⚠️ Error deleting runs/: {e}")

atexit.register(cleanup_runs)

if __name__ == '__main__':
    app.run(debug=True)
