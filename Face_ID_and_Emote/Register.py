import cv2
import torch
import pandas as pd
import numpy as np
import os
from facenet_pytorch import MTCNN, InceptionResnetV1

def extract_features(img, model, mtcnn):
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img)
    if img_cropped is None:
        return None
    img_cropped = img_cropped.to("cuda")
    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = model(img_cropped.unsqueeze(0)).detach().cpu().numpy()
    return img_embedding

def extract_features_and_save(imgpath, model, mtcnn, csv_path, name):
    img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
    features = extract_features(img, model, mtcnn)
    df = pd.DataFrame(features)
    df['Name'] = name
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

def main():
    image_path = ''
    csv_path = ''
    name = ''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, post_process=True, device = device) # Define MTCNN module
    model = InceptionResnetV1(pretrained='vggface2',device=device).eval()
    extract_features_and_save(image_path, model, mtcnn, csv_path, 'Name')

if __name__ == '__main__':
    main()