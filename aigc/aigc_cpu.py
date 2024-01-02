from . import aigc_base as base
import torch

class AigcCPU(base.AigcBase):

    def __createPipeline(self, style):
        txt2ImagePipeline, img2imgPipeline = super().__createPipeline(style)
        return self.__configPiplelines(txt2ImagePipeline, img2imgPipeline)
    
    def __createGenerator(self):
        if self.seed != None:
            generator = torch.Generator('cpu').manual_seed(self.seed)
            return generator
        return None

    def __configPiplelines(self, txt2ImagePipeline, img2imgPipeline):
        if txt2ImagePipeline:
            txt2ImagePipeline.to("cpu")

        if img2imgPipeline:
            img2imgPipeline.to("cpu")

        return [txt2ImagePipeline, img2imgPipeline]

