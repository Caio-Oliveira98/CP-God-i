#código-fonte inspirado nesse repositório https://github.com/arunponnusamy/cvlib
import concurrent.futures
from cvlib.object_detection import draw_bbox
import numpy as np
import urllib.request
import cvlib as cv
import cv2
import matplotlib.pyplot as plt

url='http://192.168.3.103/cam-hi.jpg'
im=None
im_saida=None
 
def run1():
    cv2.namedWindow("Transmissao ao Vivo", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        im = cv2.imdecode(imgnp,-1)
 
        cv2.imshow('Transmissao ao Vivo',im)
        key=cv2.waitKey(5)
        if key==ord('q'):
            break
            
    cv2.destroyAllWindows()
        
def run2():
    cv2.namedWindow("Deteccao", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        im = cv2.imdecode(imgnp,-1)
 
        bbox, label, conf = cv.detect_common_objects(im, confidence=0.6, model='yolov3-tiny')
        im_saida = draw_bbox(im, bbox, label, conf)
 
        cv2.imshow('Deteccao',im_saida)
        key=cv2.waitKey(5)
        if key==ord('q'):
            break
            
    cv2.destroyAllWindows()
 
 
 
if __name__ == '__main__':
    print("")
    print("Iniciado... Ignore mensagens de erro do <cudart>")
    print("")
    with concurrent.futures.ProcessPoolExecutor() as executer:
            #f1= executer.submit(run1)
            f2= executer.submit(run2)