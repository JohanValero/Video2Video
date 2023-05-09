import os
import torch
import base64
import io
import numpy as np

from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify

app = Flask(__name__)

cOBJECT_DETECTOR_STATUS = False
cCLIP_STATUS = False
cOPEN_POSE_STATUS = False
cMLSD_STATUS = True
cSAM_STATUS = False
cMLSD_CONTROLNET_STATUS = False

def decode_np_array(iEncodedArray):
    vArray = iEncodedArray.encode('utf-8')
    vArray = base64.b64decode(vArray)
    vBuffer = io.BytesIO(vArray)
    return np.load(vBuffer)

def encode_np_array(iArray : np.array):
    vBuffer = io.BytesIO()
    np.save(vBuffer, iArray)
    vBuffer.seek(0)
    return base64.b64encode(vBuffer.read()).decode('utf-8')

def generate_json_response(iJson):
    vResponse = jsonify(iJson)
    vResponse.headers.add('Access-Control-Allow-Origin', '*')
    return vResponse
@app.route("/", methods = ["GET"])
def home():
    try:
        vJson = {"message": "¡Server funcionando correctamente!"}
        return generate_json_response(vJson)
    except Exception as ex:
        print(ex)
        raise ex

if cOBJECT_DETECTOR_STATUS:
    from transformers import DetrImageProcessor, DetrForObjectDetection

    print("\t> Loading model for Object detector.")
    cOBJECT_DETECTOR_DEVICE = os.getenv("MODEL_DEVICE", "cuda:0")
    cOBJECT_DETECTOR_PROCESSOR = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
    cOBJECT_DETECTOR_MODEL = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
    cOBJECT_DETECTOR_MODEL = cOBJECT_DETECTOR_MODEL.to(cOBJECT_DETECTOR_DEVICE)
    print(f"\t> Object detector model loaded in device {cOBJECT_DETECTOR_DEVICE}.")

    @app.route("/genera/object_detection", methods = ["POST"])
    def detect_object():
        try:
            vRequestJson = request.get_json()
            vImage = vRequestJson['image']
            vThreshold = vRequestJson['threshold']
            vThreshold = vThreshold if vThreshold is not None else 0.9
            vImage = decode_np_array(vImage)

            vModelInput = cOBJECT_DETECTOR_PROCESSOR(
                    images = vImage,
                    return_tensors = "pt"
                ).to(cOBJECT_DETECTOR_DEVICE)
            vModelOutput = cOBJECT_DETECTOR_MODEL(**vModelInput)
            
            vTargetSizes = torch.tensor([vImage.shape[:2]])
            vResults = cOBJECT_DETECTOR_PROCESSOR.post_process_object_detection(
                    vModelOutput,
                    target_sizes = vTargetSizes,
                    threshold = vThreshold
                )[0]
            vResultResponse = []
            for tTorchScore, tTorchLabel, tTorchBox in zip(vResults["scores"], vResults["labels"], vResults["boxes"]):
                vResult = {
                    "score": tTorchScore.item(),
                    "label": tTorchLabel.item(),
                    "label_name": cOBJECT_DETECTOR_MODEL.config.id2label[tTorchLabel.item()],
                    "box": tTorchBox.tolist()
                }
                vResultResponse.append(vResult)
            vJsonResponse = {
                'result': 'success',
                'data': vResultResponse
            }
            return generate_json_response(vJsonResponse)
        except Exception as ex:
            print(ex)
            raise ex

if cCLIP_STATUS:
    from transformers import CLIPProcessor, CLIPModel

    print("\t> Loading CLIP server.")
    cCLIP_DEVICE = os.getenv("MODEL_DEVICE", "cpu")
    cCLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    cCLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    cCLIP_MODEL = cCLIP_MODEL.to(cCLIP_DEVICE)
    print(f"\t> CLIP model loaded in device {cCLIP_DEVICE}.")
    
    @app.route("/genera/clip", methods = ["POST"])
    def clip_process():
        try:
            vRequestJson = request.get_json()
            vImage = vRequestJson['image']
            vImage = decode_np_array(vImage)
            vTags  = vRequestJson['tags']

            vInputs = cCLIP_PROCESSOR(
                text = vTags,
                images = vImage,
                return_tensors = "pt",
                padding = True
            )
            vInputs.to(cCLIP_DEVICE)

            vOutPuts = cCLIP_MODEL(** vInputs)
            vLogits_per_image = vOutPuts.logits_per_image       # this is the image-text similarity score
            vProbabilitys = vLogits_per_image.softmax(dim = 1)  # we can take the softmax to get the label probabilities

            vJsonResponse = {
                'result': 'success',
                'data': vProbabilitys.tolist()[0]
            }
            return generate_json_response(vJsonResponse)
        except Exception as ex:
            print(ex)
            raise ex

if cOPEN_POSE_STATUS:
    from controlnet_aux import OpenposeDetector

    print("\t> Loading model for open pose detector.")
    gOPEN_POSE_DETECTOR = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    print(f"\t> Open pose model loaded.")

    @app.route("/genera/open_pose", methods = ["POST"])
    def detect_pose():
        try:
            vRequestJson = request.get_json()
            vImage = vRequestJson['image']
            vImage = decode_np_array(vImage)
            vImage = gOPEN_POSE_DETECTOR(vImage)
            vJsonResponse = {
                'result': 'success',
                'data': encode_np_array(np.array(vImage))
            }
            return generate_json_response(vJsonResponse)
        except Exception as ex:
            print(ex)
            raise ex

if cMLSD_STATUS:
    from controlnet_aux import MLSDdetector

    print("\t> Loading model for MLS lines.")
    gMLS_DETECTOR = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    print(f"\t> MLS model loaded.")

    @app.route("/genera/mlsd", methods = ["POST"])
    def detect_mlsd():
        try:
            vRequestJson = request.get_json()
            vImage = vRequestJson['image']
            vImage = decode_np_array(vImage)
            vImage = gMLS_DETECTOR(vImage)
            vJsonResponse = {
                'result': 'success',
                'data': encode_np_array(np.array(vImage))
            }
            return generate_json_response(vJsonResponse)
        except Exception as ex:
            print(ex)
            raise ex

if cSAM_STATUS:
    from segment_anything import sam_model_registry, SamPredictor

    print("Loading SAM server.")
    cSAM_DEVICE = os.getenv("MODEL_DEVICE", "cpu")
    cSAM_MODEL_TYPE, cSAM_CHECKPOINT = "vit_h", "../resources/models/sam_vit_h_4b8939.pth" # 2.5G
    #cSAM_MODEL_TYPE, cSAM_CHECKPOINT = "vit_l", "../resources/models/sam_vit_l_0b3195.pth" # 1.2 GB
    #cSAM_MODEL_TYPE, cSAM_CHECKPOINT = "vit_b", "../resources/models/sam_vit_b_01ec64.pth" # 300 MB
    cSAM_MODEL = sam_model_registry[cSAM_MODEL_TYPE](checkpoint = cSAM_CHECKPOINT)
    cSAM_MODEL.to(device = cSAM_DEVICE)
    cSAM_PREDITOR = SamPredictor(cSAM_MODEL)
    print(f"\t> SAM loaded in device {cSAM_DEVICE}.")

    @app.route("/genera/sam/boxes", methods = ["POST"])
    def sam_boxes():
        try:
            vRequestJson = request.get_json()
            vBoxes = vRequestJson['boxes']
            vImage = vRequestJson['image']
            vImage = decode_np_array(vImage)
            
            vSamBoxes = torch.tensor(vBoxes, device = cSAM_PREDITOR.device)
            vSamBoxes = cSAM_PREDITOR.transform.apply_boxes_torch(vSamBoxes, vImage.shape[:2])
            
            cSAM_PREDITOR.set_image(vImage)
            vMasks, _, _ = cSAM_PREDITOR.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = vSamBoxes,
                multimask_output = False,
            )
            
            vMasks = [x.cpu().numpy()    for x in vMasks] # Obtain the masks in the GPU to Numpy Arrays.
            vMasks = [np.squeeze(x)      for x in vMasks] # Modify the shape (1, Width, Height) to (Width, Height).
            vMasks = [encode_np_array(x) for x in vMasks]

            vJsonResponse = {
                'result': 'success',
                'data': vMasks
            }
            return generate_json_response(vJsonResponse)
        except Exception as ex:
            print(ex)
            raise ex

if cMLSD_CONTROLNET_STATUS:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

    print("\t> Loading MLSD ControlNet server.")
    gMLSD_CONTROLNET = ControlNetModel.from_pretrained(
      "lllyasviel/sd-controlnet-mlsd",
      torch_dtype = torch.float16
    )
    gMLSD_STABLE_DIFUSSION = StableDiffusionControlNetPipeline.from_pretrained(
      "runwayml/stable-diffusion-v1-5",
      controlnet = gMLSD_CONTROLNET,
      safety_checker = None,
      torch_dtype=torch.float16
    )
    gMLSD_STABLE_DIFUSSION.scheduler = UniPCMultistepScheduler.from_config(gMLSD_STABLE_DIFUSSION.scheduler.config)
    #pipe.enable_xformers_memory_efficient_attention()
    gMLSD_STABLE_DIFUSSION.enable_model_cpu_offload()
    print(f"\t> MLSD Control net loaded in device CUDA (FORCED).")
  
    @app.route("/genera/controlnet/mlsd", methods = ["POST"])
    def generate_controlnet_mlsd():
        try:
            vRequestJson = request.get_json()
            vPromt = vRequestJson['prompt']
            vImage = vRequestJson['image']
            vImage = decode_np_array(vImage)
            vImage = Image.fromarray(vImage)
            vImage = gMLSD_STABLE_DIFUSSION(
                    vPromt,
                    vImage,
                    num_inference_steps = 20
                ).images[0]
            vJsonResponse = {
                'result': 'success',
                'data': encode_np_array(np.array(vImage))
            }
            return generate_json_response(vJsonResponse)
        except Exception as ex:
            print(ex)
            raise ex

if __name__ == "__main__":
    cPORT = os.getenv("PORT", 88)
    print(f"\t> Running the server in development mode in port {cPORT}.")
    app.run(port = cPORT, host = "0.0.0.0", debug = False)