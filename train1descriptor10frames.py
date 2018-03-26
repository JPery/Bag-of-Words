from scipy.cluster.vq import vq, kmeans
import cv2
import pickle
import numpy as np
from functions import *

orb = cv2.ORB_create()

list_videos = ['videos/arbol.webm', 'videos/casa.webm', 'videos/pez.webm', 'videos/zagal.webm']
all_descriptors = []
for video in list_videos:
    capture = cv2.VideoCapture(video)
    counter = 0
    while True:
        # Capturamos,
        ret, frame = capture.read()
        if ret == False:
            break
        # Comprobamos que el frame es el numero 10
        if counter < 10:
            counter += 1
            continue
        
        # Escalamos,
        recortada = frame[100:-100, 100:-170]
	recortada_gris = cv2.cvtColor(recortada, cv2.COLOR_BGR2GRAY)
	resize_ratio = float(1120.0 / float(len(recortada_gris)))
	resized = cv2.resize(recortada_gris, (0,0), fx=resize_ratio, fy=resize_ratio)
        all_bounds = extract_rois(resized)
	keypoints = []
	for i in range(len(all_bounds)):
                y1, y2 = (all_bounds[i][0]-10, all_bounds[i][0]+all_bounds[i][2]+10)
                x1, x2 = (all_bounds[i][1]-10, all_bounds[i][1]+all_bounds[i][3]+10)
                roi = resized[x1:x2, y1:y2]
    		kp, descriptors = orb.detectAndCompute(roi, None)
                if descriptors is not None and len(descriptors) > 20:
                    all_descriptors.extend(descriptors)
                    keypoints.extend(kp)
    

whitened, dev = whiten(np.array(all_descriptors))
book, distortion = kmeans(whitened, 20)
pickle.dump(dev, open('data/dev1every10.pickle', 'w'))
pickle.dump(book, open('data/codebook1every10.pickle', 'w'))
pickle.dump(distortion, open('data/codebook_distortion1every10.pickle', 'w'))

for video in list_videos:
    video_histograms = []
    capture = cv2.VideoCapture(video)
    counter = 0
    while True:
        # Capturamos,
        ret, frame = capture.read()
        if ret == False:
            break
        if counter < 10:
            counter += 1
            continue
        # Escalamos,
        recortada = frame[100:-100, 100:-170]
	recortada_gris = cv2.cvtColor(recortada, cv2.COLOR_BGR2GRAY)
	resize_ratio = float(1120.0 / float(len(recortada_gris)))
	resized = cv2.resize(recortada_gris, (0,0), fx=resize_ratio, fy=resize_ratio)
        all_bounds = extract_rois(resized)
        for i in range(len(all_bounds)):
                y1, y2 = (all_bounds[i][0]-10, all_bounds[i][0]+all_bounds[i][2]+10)
                x1, x2 = (all_bounds[i][1]-10, all_bounds[i][1]+all_bounds[i][3]+10)
                roi = resized[x1:x2, y1:y2]
                # Sacamos keypoints y descriptores,
    		kp, descriptors = orb.detectAndCompute(roi, None)
                if descriptors is not None and len(descriptors) > 20:
                    histograma, limites = np.histogram(vq(descriptors/dev, book)[0], bins=range(20), density=True )
                    video_histograms.append(histograma)

    pickle.dump(video_histograms, open('data/' + video.rsplit('.', 1)[0] + '1every10.pickle', 'w'))
