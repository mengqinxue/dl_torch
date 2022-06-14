## Pytorch Journey

archieve a model
```commandline
torch-model-archiver --model-name dm1 --version 1.0 --serialized-file pretrained_models/dummy_model.p
th --handler dummy_model_handler
```

start serve and host the model
```commandline
torchserve --model-store model_store
```

register your model 
```commandline
curl -X POST "http://localhost:8081/models?url=dm1.mar"
```

https://markdowner.net/article/238437127418474496