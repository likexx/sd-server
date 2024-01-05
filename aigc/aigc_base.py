import base64
from io import BytesIO
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DiffusionPipeline
from diffusers.pipelines.stable_diffusion import safety_checker

from . import aigc_base, models


def remove_nsfw_check(self, clip_input, images) :
    return images, [False for i in images]

class AigcBase:
    modelMap = models.MODELS

    def __init__(self, params):
        self.prompt = params.prompt
        self.negPrompt = params.negPrompt
        self.image = params.image
        self.steps = params.steps
        self.numImages = params.numImages
        self.imageHeight = params.imageHeight
        self.imageWidth = params.imageWidth
        self.style = params.style
        self.seed = params.seed

    def __loadFromExistingModels(self, model, requireSafetyChecker):
        if not requireSafetyChecker:
            pipeline = StableDiffusionPipeline.from_single_file(model, safety_checker = None, requires_safety_checker = False)
        else:
            pipeline = StableDiffusionPipeline.from_single_file(model)
        components = pipeline.components
        img2imgPipeline = StableDiffusionImg2ImgPipeline(**components)     
        return [pipeline, img2imgPipeline]
    
    def __loadForAutoPipeline(self, model, requireSafetyChecker):
        print('load for {}'.format(model))
        if not requireSafetyChecker:
            # pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
            #                                                 # torch_dtype=torch.float16, 
            #                                                 use_safetensors=True, 
            #                                                 variant="fp16", 
            #                                                 safety_checker = None, 
            #                                                 requires_safety_checker = False
            #                                                 )
            pipeline = AutoPipelineForText2Image.from_pretrained(
                            model, 
                            # torch_dtype=torch.float16, 
                            variant="fp16",
                            use_safetensors=True,
                            safety_checker = None,
                            requires_safety_checker = False
                        )
        else:
            # pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
            #                                                 # torch_dtype=torch.float16, 
            #                                                 use_safetensors=True, 
            #                                                 variant="fp16"
            #                                                 )
            pipeline = AutoPipelineForText2Image.from_pretrained(
                            model, 
                            # torch_dtype=torch.float16, 
                            variant="fp16",
                            use_safetensors=True,
                        )
        
        img2imgPipeline = AutoPipelineForImage2Image.from_pipe(pipeline)

        # components = pipeline.components
        # img2imgPipeline = StableDiffusionXLImg2ImgPipeline(**components)

        return [pipeline, img2imgPipeline]
    
    def __loadFromOnlineModels(self, model, requireSafetyChecker):
        if not requireSafetyChecker:
            pipeline = StableDiffusionPipeline.from_pretrained(model, 
                                                                revision="fp16", 
                                                                torch_dtype=torch.float16, 
                                                                safety_checker = None, 
                                                                requires_safety_checker = False)
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(model, 
                                                                revision="fp16", 
                                                                torch_dtype=torch.float16)

        components = pipeline.components
        img2imgPipeline = StableDiffusionImg2ImgPipeline(**components)     

        return [pipeline, img2imgPipeline]


    def __createPipeline(self, style):
        print("******* create pipeline for style {}".format(style))
        modelMap = AigcBase.modelMap
        if not style in modelMap:
            print("invalid style: " + style)
            print("fallback to use anything (cartoon)")
            style = "cartoon"

        modelConfig = modelMap[style]
        requireSafetyChecker = True
        if modelConfig.get('nsfw', True) == False:
            # edit StableDiffusionSafetyChecker class so that, when called, it just returns the images and an array of True values
            print("remove nsfw check")
            safety_checker.StableDiffusionSafetyChecker.forward = remove_nsfw_check
            requireSafetyChecker = False

        model = modelConfig["model"]
        lora = modelConfig.get("lora", None)

        print("using model {}".format(model))
        if model.endswith('.safetensors') or model.endswith('.ckpt'):
            pipeline, img2imgPipeline = self.__loadFromExistingModels(model, requireSafetyChecker)    
        elif modelConfig['pipeline_type']=='auto':
            pipeline, img2imgPipeline = self.__loadForAutoPipeline(model, requireSafetyChecker)    
        else:
            pipeline, img2imgPipeline = self.__loadFromOnlineModels(model, requireSafetyChecker)

        if lora:
            if lora.endswith('.safetensors'):
                print("load lora weights")
                pipeline.load_lora_weights(".", weight_name=lora)
            else:
                pipeline.unet.load_attn_procs(lora)

        return [pipeline, img2imgPipeline]

    def __configPiplelines(self, txt2ImagePipeline, img2imgPipeline):
        pass

    def __createGenerator(self):
        pass
    
    def generate(self):
        result = []

        modelConfig = self.modelMap.get(self.style, None)
        if not modelConfig:
            print('ERROR: cannot load model {}'.format(self.style))
            return result
    
        enhancedPrompt = self.prompt
        promptSuggestions = modelConfig.get("prompts", [])

        for w in promptSuggestions:
            if enhancedPrompt.find(w) < 0:
                enhancedPrompt += "," + w

        txt2imgPipeline, img2imgPipeline = self.__createPipeline(self.style)
        
        generator = self.__createGenerator()

        if not self.image or not img2imgPipeline:
            print("generate with txt2img")
            images = txt2imgPipeline(enhancedPrompt,
                                negative_prompt=self.negPrompt,
                                num_images_per_prompt=self.numImages,
                                num_inference_steps=self.steps,
                                height=self.imageHeight,
                                width=self.imageWidth,
                                generator=generator).images
        else:
            print("generate with img2img")
            init_image = self.__base64ToRGB(self.image)
            init_image = init_image.resize((self.imageWidth, self.imageHeight))
            images = img2imgPipeline(enhancedPrompt,
                                    image=init_image,
                                    negative_prompt=self.negPrompt,
                                    num_images_per_prompt=self.numImages,
                                    num_inference_steps=self.steps,
                                    generator = generator
                                    ).images
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            # finalImage.save(buffered, format="JPEG")
            base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            result.append({'base64_data': base64_str})

        return result

    def __base64ToRGB(self, base64_data):
        # Decode the base64 data
        decoded_data = base64.b64decode(base64_data)
        
        # Convert the decoded data to an image
        img_buffer = BytesIO(decoded_data)
        img = Image.open(img_buffer)
        
        # Convert to RGB
        rgb_img = img.convert('RGB')
        
        return rgb_img
