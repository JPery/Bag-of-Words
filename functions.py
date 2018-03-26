import cv2

def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return (0,0,0,0)
  return (x, y, w, h)

def union_over_intersection(a, b, threshold=1000):
    x, y, w, h = intersection(a, b)
    intersect_area = w*h
    return intersect_area > threshold

def whiten(v, dev=None):
    v2 = v.astype(np.float64)
    if type(dev) == type(None):
        dev = np.std(a=v2, axis=0)
    ret = np.array(v2).astype(np.float64)
    row = 0
    while row < len(ret):
        column = 0
        while column < len(ret[row]):
            if dev[column] > 0.0000001:
                ret[row][column] /= dev[column]
            else:
                dev[column] = 0.
            column += 1
        row += 1
    return ret, dev

def aplicarFiltrosMorfologicos(imagen):
        dilatada = cv2.dilate(imagen,None)
        erosionada = cv2.erode(dilatada, None, iterations=2)
	return cv2.dilate(erosionada,None, iterations=4)
    

def extract_rois(resized, overlap=False):
        binaria = 255-cv2.inRange(resized, 90, 250)
        binaria = aplicarFiltrosMorfologicos(binaria)
        i, contornos, h = cv2.findContours(binaria, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	all_bounds = []
	for contorno in contornos:
            if cv2.contourArea(contorno) > 3000:
                if overlap:
                    i = 0
                    selected_bounds = None
                    while selected_bounds is None and i < len(all_bounds):
                        bound = all_bounds[i]
                        if union_over_intersection(bound, cv2.boundingRect(contorno)):
                            selected_bounds = bound
                        i+=1
                    if selected_bounds is None:
                        bounds = cv2.boundingRect(contorno)
                        all_bounds.append(bounds)
                    else:
                        all_bounds.remove(selected_bounds)
                        all_bounds.append(union(selected_bounds, cv2.boundingRect(contorno)))
                else:
                    bounds = cv2.boundingRect(contorno)
                    all_bounds.append(bounds)
        return all_bounds


def checkNN(histograma):
    distance = float('inf')
    computed_class = 'Desconocido'
    for clase, descriptor_class in clases_histogramas.items():
            for histogram in descriptor_class:
                new_distance = spatial.distance.euclidean(histograma, histogram)
                if new_distance < distance:
                    distance = new_distance
                    computed_class = clase
    return computed_class