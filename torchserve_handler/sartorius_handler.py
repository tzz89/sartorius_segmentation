import detectron2.data.transforms as T
import torch
import os
import logging
import base64
from PIL import Image, ImageDraw
import io
import numpy as np
import random

logger = logging.getLogger(__name__)


class SatoriusHandler():
    def initialize(self, context):
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available(
        ) and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # Expects torchscripted file

        logger.debug("Loading torchscript model")
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = self._load_torchscript_model(model_pt_path)
        self.initialized = True

    def _load_torchscript_model(self, model_pt_path):
        """Loads the PyTorch model and returns the NN model object.
        Args:
            model_pt_path (str): denotes the path of the model file.
        Returns:
            (NN Model Object) : Loads the model object.
        """
        return torch.jit.load(model_pt_path, map_location=self.device)

    def preprocess(self, row):
        # converts a input row to image

        image = row.get("data") or row.get("body")
        if isinstance(image, str):
            image = base64.b64decode(image)
        if isinstance(image, (bytearray, bytes)):
            raw_image = Image.open(io.BytesIO(image))

        # (H W) #since it is grayscale BGR and RGB is the same
        image = np.array(raw_image)
        # convert to 3 channels
        image = np.stack([raw_image, raw_image, raw_image], axis=2)

        augmentation = T.ResizeShortestEdge([800, 800], 1333)
        aug = augmentation.get_transform(image)
        image = aug.apply_image(image)

        image = torch.as_tensor(image.astype(
            "float32").transpose(2, 0, 1))  # (CHW)

        return image, aug, raw_image

    def postprocess(self, prediction, augmentation, pil_image):
        """ This post processing function, will draw the bboxes and 
        segmentation map and and return the base64 encoded image """

        inv_augmentation = augmentation.inverse()
        bboxes = inv_augmentation.apply_box(prediction[0].detach())
        masks = prediction[2].detach().numpy()

        rgbimg = Image.new("RGBA", pil_image.size)  # creating a RGB image
        rgbimg.paste(pil_image)

        # creating an image of half the transparency
        original_rgbimg = rgbimg.copy()
        original_rgbimg.putalpha(128)

        draw = ImageDraw.Draw(rgbimg)

        for bbox, mask in zip(bboxes, masks):
            random_hex = "#" + \
                ''.join([random.choice('ABCDEF0123456789') for i in range(6)])
            draw.rectangle(
                [int(coord) for coord in bbox],
                width=1,
                outline=random_hex
            )
            bbox_width = int(bbox[2]-bbox[0])
            bbox_height = int(bbox[3] - bbox[1])

            mask_map = Image.fromarray(np.squeeze(mask), mode='F')
            mask_map = mask_map.resize(
                (bbox_width, bbox_height), resample=Image.BILINEAR)
            mask_map_array = np.array(mask_map) > 0.5
            mask_map = Image.fromarray(mask_map_array)

            draw.bitmap((int(bbox[0]), int(bbox[1])),
                        mask_map, fill=random_hex)

        rgbimg.putalpha(128)
        combined_image = Image.alpha_composite(original_rgbimg, rgbimg)

        # converting to base64
        buffer = io.BytesIO()
        combined_image.save(buffer, format='png')
        img_str = base64.b64encode(buffer.getvalue())
        buffer.close()

        return img_str

    def handle(self, data, context):
        retvals = []
        for row in data:
            image, augmentation, pil_image = self.preprocess(row)
            image = image.to(self.device)
            with torch.no_grad():
                prediction = self.model(image)
                retvals.append(self.postprocess(
                    prediction, augmentation, pil_image))
        return retvals
