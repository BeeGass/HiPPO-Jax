from typing import Union

def preprocess(input : Union[str, Image, Video, Audio]) -> Tensor:
    # implementation

def inference(input : Tensor) -> Tensor:
    # implementation

def postprocess(input : Tensor) -> Union[str, Image, Video, Audio]:
    # implementation