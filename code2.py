from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import cv2
from PIL import Image, ImageDraw, ImageFont
import urllib


font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.7
fontColor              = (0,0,255)
lineType               = 2
credentials = ApiKeyCredentials(in_headers={"Prediction-key": "3e279055ef9f43d39045930eed2312d5"})
predictor = CustomVisionPredictionClient("https://southcentralus.api.cognitive.microsoft.com/", credentials)

url = "https://img.europapress.es/fotoweb/fotonoticia_20201223093019_420.jpg" 
urllib.request.urlretrieve(url, "python.png")
imagen=cv2.imread("python.png")
height, width, channels = imagen.shape
Resultado = predictor.detect_image_url("86693ba7-c702-40b8-a61b-beadd9e115b7", "Iteration3", url) 

for prediction in Resultado.predictions:
    if prediction.probability > 0.4:
        bbox = prediction.bounding_box
        tag = prediction.tag_name
        probabilidad= int(prediction.probability * 100)
        result_image = cv2.rectangle(imagen, (int(bbox.left * width), int(bbox.top * height)), (int((bbox.left + bbox.width) * width), int((bbox.top + bbox.height) * height)), (0, 255, 0), 3)
        bottomLeftCornerOfText = (int(bbox.left*width),int(((bbox.top*height)+(bbox.height*height))))
        cv2.putText(result_image,str(probabilidad)+"% "+tag,
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
        cv2.imwrite('result.png', result_image)     
cv2.imshow('Resultado',result_image)
cv2.waitKey(0)




        
