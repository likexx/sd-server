class AigcParam:
    NUM_IMAGES = 4
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    INFER_STEPS = 50
    NEGATIVE_PROMPT="(worst quality, low quality, normal quality:1.4), lowres, bad anatomy, ((bad hands)), text, error, missing fingers, extra digit, fewer digits,head out of frame, cropped, letterboxed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, censored, letterbox, blurry, monochrome, fused clothes, nail polish, boring, extra legs, fused legs, missing legs, missing arms, extra arms, fused arms, missing limbs, mutated limbs, dead eyes, empty eyes, 2girls, multiple girls, 1boy, 2boys, multiple boys, multiple views, jpeg artifacts, text, signature, watermark, artist name, logo, low res background, low quality background, missing background, white background,deformed"

    def __init__(
            self,
            prompt, 
            negPrompt = NEGATIVE_PROMPT, 
            image = None, 
            steps = INFER_STEPS,
            numImages = NUM_IMAGES,
            imageHeight = IMG_HEIGHT,
            imageWidth = IMG_WIDTH,
            style = "cartoon",
            seed = None
    ):
        self.prompt = prompt
        self.negPrompt = negPrompt
        self.image = image
        self.steps = steps
        self.numImages = numImages
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.style = style
        self.seed = seed
        