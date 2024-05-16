'''
create by wyattPol
'''
import torch
import numpy as np
from network2 import Net 
import cv2
import os

model = Net()  
checkpoint = torch.load('saved_models/500_epoch_v8_checkpoint.pth.tar') 
model.load_state_dict(checkpoint)  
model.eval()

folder_path = 'datasets/test/'
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (640, 640))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        gray_image_np = np.array(gray_image)
        image_tensor = torch.from_numpy(gray_image_np).unsqueeze(0).unsqueeze(0).float()  

        with torch.no_grad():
            output = model(image_tensor)  # Forward pass

        predicted_bbox = output.squeeze().tolist()
        center_x, center_y, width, height = predicted_bbox
        xmin = int(center_x - width / 2)
        ymin = int(center_y - height / 2)
        xmax = int(center_x + width / 2)
        ymax = int(center_y + height / 2)

        print(f"xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")
        image_with_bbox = cv2.rectangle(resized_image.copy(), (xmin, ymin), (xmax, ymax), (0, 255, 255), 3)
        cv2.imshow(f'Image {filename} with Bounding Box', image_with_bbox)
        cv2.waitKey(0)
cv2.destroyAllWindows()
