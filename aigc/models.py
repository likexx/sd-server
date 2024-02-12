
localAnythingAndEverythingPath = "/Users/likezhang/projects/models/anythingAnd_anythingAndEverything.safetensors"
stableDiffustion1_5 = "runwayml/stable-diffusion-v1-5"
stableDiffuxionXL = "stabilityai/stable-diffusion-xl-base-1.0"
stableDiffuxionTurbo = "stabilityai/sdxl-turbo"
hardcoreAsianPornPath = "/Users/likezhang/projects/models/hardcoreAsianPorn_v20.safetensors"
hardcoreHentaiPath = "/Users/likezhang/projects/models/hardcoreHentai12_v12BakedVAE.safetensors"
grapfruitHentaiPath = "/Users/likezhang/projects/models/grapefruitHentaiModel_grapefruitv41.safetensors"
likezhangPath = "/home/likezhang/output"

DEFAULT_PROMPT_SUGGESTION = [
    'best quality',
    'realistic',
    'masterpiece',
    'vivid',
    'vibrant colors',
    'photorealistic',
]

CARTOON_PROMPT_SUGGESTION = [
    'Japanese manga', 
    'cartoon', 
    'comics', 
    'animation',
    'best quality',
    'details',
    'masterpiece',
    'vivid',
    'vibrant colors',
]

MODELS = {
    "cartoon": { "model": localAnythingAndEverythingPath, "nsfw": False, "prompts": CARTOON_PROMPT_SUGGESTION },
    "cartoon-adult": { "model": localAnythingAndEverythingPath, "nsfw": False, "prompts": CARTOON_PROMPT_SUGGESTION },

    "real": {"model": stableDiffuxionXL, "compel": "2", "pipeline_type": "auto", "nsfw": False, "prompts": DEFAULT_PROMPT_SUGGESTION},
    "real-adult": {"model": hardcoreAsianPornPath, "nsfw": False, "prompts": DEFAULT_PROMPT_SUGGESTION },

    "turbo": {"model": stableDiffuxionTurbo, "compel": "2", "pipeline_type": "auto", "nsfw": False, "prompts": DEFAULT_PROMPT_SUGGESTION},
    "turbo-cartoon": {"model": stableDiffuxionTurbo, "compel": "2", "pipeline_type": "auto", "nsfw": False, "prompts": CARTOON_PROMPT_SUGGESTION},

    "cartoon2": { "model": hardcoreHentaiPath, "nsfw": False,  "prompts": CARTOON_PROMPT_SUGGESTION },
    "cartoon2-nsfw": { "model": hardcoreHentaiPath, "nsfw": False, "prompts": CARTOON_PROMPT_SUGGESTION },

    "anim-porn": { "model": grapfruitHentaiPath, "nsfw": False, "prompts": CARTOON_PROMPT_SUGGESTION },
    "likezhang": { "model": likezhangPath, "pipeline_type": "auto", "nsfw": False, "prompts": DEFAULT_PROMPT_SUGGESTION }
}
