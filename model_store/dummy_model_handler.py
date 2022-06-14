import os
import torch
from pandas._libs import properties
from ts.torch_handler.base_handler import BaseHandler
from src.models.dummy_model import DummyModel


class DummyModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self._context = None
        self.initialized = False
        self.model = DummyModel()
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        print(self.device)

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: context contains model server system properties
        :return:
        """
        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = torch.jit.load(model_pt_path)
        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")
        return preprocessed_data

    def inference(self, data, *args, **kwargs):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        pred_out = self.model.forward(data)
        return pred_out

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        print('Post-process')
        postprocess_output = inference_output
        return postprocess_output

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)

