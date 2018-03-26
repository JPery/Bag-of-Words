import cv2
import numpy as np
import pickle
from scipy import spatial
from scipy.cluster.vq import vq, kmeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from functions import *

book = pickle.load(open('data/codebook1every10.pickle'))
clases = ['arbol', 'casa', 'pez', 'zagal']
clases_histogramas = {}


dev = pickle.load(open('data/dev1every10.pickle'))

for clase in clases:
    clases_histogramas[clase] = pickle.load(open('data/' + clase + '1every10.pickle'))

X = []
Y = []
for key, value in clases_histogramas.items():
    X.extend(value)
    Y.extend([key]*len(value))

predictor = KNeighborsClassifier()

predictor.fit(X,Y)

    
def detectar_clase(gray_image):
    # Sacamos keypoints y descriptores
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    if descriptors is not None:
        histograma, limites = np.histogram(vq(descriptors/dev, book)[0], bins=range(20), density=True )
        prediction = predictor.predict([histograma])[0]
        probabilities = predictor.predict_proba([histograma])[0]
        probability = probabilities[predictor.classes_.tolist().index(prediction)]
        # Comprobamos la probabilidad de la clase detectada
        if probabilities[predictor.classes_.tolist().index(prediction)] > 0.5:
            return prediction + ' ' + str(probability*100) + '%'
        # Si es menor del 50%, le asignamos "Desconocido"
        else:
            return 'Desconocido'
        return checkNN(histograma)
    return 'Desconocido'


# Instanciamos el detector ORB
orb = cv2.ORB_create()
# Cargamos el video de prueba
capture = cv2.VideoCapture('videos/test.webm')
while True:
        # Leemos un frame del video
	ret, frame = capture.read()
	if not ret:
		break
	recortada = frame[100:-100, 100:-170]
	recortada_gris = cv2.cvtColor(recortada, cv2.COLOR_BGR2GRAY)
	resize_ratio = float(1120.0 / float(len(recortada_gris)))
	resized = cv2.resize(recortada_gris, (0,0), fx=resize_ratio, fy=resize_ratio)
        
	all_bounds = extract_rois(resized, overlap=True)
        
        img_to_show = cv2.resize(recortada, (0,0), fx=resize_ratio, fy=resize_ratio)
        
        for i in range(len(all_bounds)):
                y1, y2 = (all_bounds[i][0]-10, all_bounds[i][0]+all_bounds[i][2]+10)
                x1, x2 = (all_bounds[i][1]-10, all_bounds[i][1]+all_bounds[i][3]+10)
                roi = resized[x1:x2, y1:y2]
    		color = (127,255,0)
    		cv2.rectangle(img_to_show, (y1,x1), (y2,x2), color, thickness=5)
    		cv2.line(img_to_show,(y1+5,x1-6), (y2-5,x1-6), color, thickness=18)
    		clase = detectar_clase(roi)
    		cv2.putText(img_to_show, clase, (y1+2,x1-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), thickness=2)
    
        wkeypoints = resized.copy()
        cv2.imshow("Arenero", img_to_show)
        #cv2.imshow("Binaria", binaria)
	cv2.waitKey(33)
