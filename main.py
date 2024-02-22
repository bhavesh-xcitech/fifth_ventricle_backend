from fastapi import FastAPI , Query ,File,UploadFile
from fastapi.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import matplotlib.pylab as plt
import numpy as np
from tensorflow import keras

import soundfile as sf
print("i am called")
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from keras import models

print("i am called2")



from typing import Optional
from pydantic import BaseModel
import os
import shutil

import librosa

import resampy



app = FastAPI()

UPLOAD_DIRECTORY = "files/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


app.mount("/static", StaticFiles(directory=UPLOAD_DIRECTORY), name="static")

class PredictionResult(BaseModel):
    Crackles: float
    Normal: float
    Pleural_Rub: float
    Wheezing: float


def extract_feature(file: UploadFile, uid: str):
    try:
        audio_data, sample_rate = librosa.load(file.file, res_type='kaiser_fast', sr=4000)

        file_name_parts = file.filename.split(".")
        image_file_name = f"{UPLOAD_DIRECTORY}/breathing/{uid}_{file_name_parts[0]}.png"
        print(image_file_name)
        plt.figure()
        plt.axis("off")
        os.makedirs(f"{UPLOAD_DIRECTORY}/breathing/", exist_ok=True)

        # librosa.display.waveshow(audio_data, )
        plt.savefig(image_file_name, transparent=True)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)

        # Save audio file
        audio_file_name = f"{UPLOAD_DIRECTORY}/breathing/{file.filename}"
        sf.write(audio_file_name, audio_data, sample_rate)

        return np.array([mfccs_scaled])
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="An error occurred while processing the file.")

async def get_breathing_type_1(file: UploadFile = File(...), uid: str = None):
    try:
        mfccs_scaled = extract_feature(file, uid)
        model_path = os.path.join(os.getcwd(), "saved_models", "saved_model")
        print(model_path)
        print("TensorFlow version:", tf.__version__)
        model = models.load_model(model_path)
        # keras.models.load_model(model_path)
        # # Reshape the MFCCs to match the model's input shape
        # mfccs_scaled = mfccs_scaled.reshape(1, -1)
        # mfccs_scaled = mfccs_scaled[np.newaxis, :]
        # print(mfccs_scaled)

        predictions = model.predict(mfccs_scaled)
        print(predictions)
        result = {
            "Crackles": float(round(predictions[0][0],4)),
            "Normal": float(round(predictions[0][1] ,4)),
            "Pleural_Rub": float(round(predictions[0][2],4)),
            "Wheezing": float(round(predictions[0][3],4))
        }
        print((result) )
        predict2 = ["Crackles", "Normal", "Pleural_Rub", "Wheezing"]
        predict2_prob = [result["Crackles"], result["Normal"], result["Pleural_Rub"], result["Wheezing"]]

        if result["Wheezing"] > 0.9:
            predict2 = ["Wheezing"]
            predict2_prob = [result["Wheezing"]]
        else:
            predict2 = ["Normal"]
            predict2_prob = [result["Normal"]]

        return {
            "result": result,
            "predict2": predict2,
            "predict2_prob": predict2_prob
        }
    except Exception as e:
        print(e)
        error_message = "An error occurred while processing the request."
        return JSONResponse(status_code=500, content={"detail": error_message})
    






    
# def predict_breathing(arr):
#   y_train_path = str(current_app.root_path) + "/saved_models/breathingYtest.pkl"
#   y_train = joblib.load(y_train_path)
#   le_path = str(current_app.root_path) + "/saved_models/breathingLabelEncoder.pkl"
#   le = joblib.load(le_path)
#   temp = []
#   count = 0
#   for i in arr:
#     y_train_index = y_train[i]
#     predict_knn_breathing = le.inverse_transform([y_train_index])
#     predict_value = predict_knn_breathing[0]
#     if predict_value not in temp:
#       temp.append(predict_value)
#       count+=1
#     if count>=2:
#       return temp
#   return temp



@app.get('/')
def root():
    return "helllo word"

@app.post("/files")
async def create_file(
    wav_file: UploadFile = File(...),
     image_file: UploadFile = File(...)
):
 
    try:
        if wav_file is None or image_file is None:
             return JSONResponse(status_code=500, content={"message": "No file provided"})
            # raise HTTPException(status_code=400, detail="No file provided")

        if not wav_file.content_type.endswith("/wav"):
             return JSONResponse(status_code=500, content={"message": "Unsupported file format. Must be an audio WAV file"})
            # raise HTTPException(status_code=415, detail="Unsupported file format. Must be an audio WAV file")
        
        if not image_file.content_type.endswith("/png"):
             return JSONResponse(status_code=500, content={"message": "Unsupported file format. Must be a PNG file"})
            # raise HTTPException(status_code=415, detail="Unsupported file format. Must be a PNG file")

        os.makedirs(f"{UPLOAD_DIRECTORY}/heart_beats", exist_ok=True)
  
        wav_file_path = os.path.join(f"{UPLOAD_DIRECTORY}/heart_beats", wav_file.filename)
        with open(wav_file_path, "wb") as buffer:
            shutil.copyfileobj(wav_file.file, buffer)
            print(wav_file_path)
            


        image_file_path = os.path.join(f"{UPLOAD_DIRECTORY}/heart_beats", image_file.filename)
        with open(image_file_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)
        wav_file_url = f"http://127.0.0.1:8000/static/heart_beats/{wav_file.filename}"
        image_file_url = f"http://127.0.0.1:8000/static/heart_beats/{image_file.filename}"

        return {
            "wav_file_url": wav_file_url,
            "image_file_url": image_file_url
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(status_code=500, content={"message": "Internal server error"})


@app.post("/breathing_analysis")
async def breathing_analysis(file: UploadFile = File(...), uid: str = None):
    image_url = "http://127.0.0.1:8000/static/"+uid+"_"+file.filename.split(".")[0]+".png"
    breathing_results = await get_breathing_type_1(file, uid)  # Await the result of the asynchronous function call
    return JSONResponse({
        "predict": breathing_results["result"],
        "image_url": image_url,
        "audio_path": str("http://127.0.0.1:8000/static/analyse/") + uid + "_" + file.filename,
        "predict2": breathing_results["predict2"],
        "predict2_prob": breathing_results["predict2_prob"]
    })


# @app.get('/item/{item_id}')
# def getItemId(item_id):
#     return f"your id id is - {item_id}"

# fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

# @app.get('/item')
# def list_item(item_num:int =0, limit:int =10):
#    return fake_items_db[item_num : item_num + limit]


# @app.get("/item/{item_id}")
# def get_items(item_id:str , q:Optional[str] = None , short:bool= True):

#     item = {"item_id":item_id ,"short": not short}
   
#     # it means if q is not none 
#     if q:
#         item.update({"q":q})
#         # return item
#     if not short:
#         item.update(
#             {
#                 "description": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut consectetur."
#             }
#         )
#     return item



# @app.post('/request_body')
# def create_item(item:Item):
#     return item

# @app.get('/string_validation')
# def stringValidation(q = Query(None)):
#     print(q)

#     result = {"item":[{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]}
#     if q:
#         print("null\n")
#         result.update({"q":q})
#     return result
#     # return result
