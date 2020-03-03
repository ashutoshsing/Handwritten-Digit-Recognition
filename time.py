
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage
from azureml.core import Workspace as ws
import score
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
import logging
from azureml.core.webservice import AciWebservice
logging.basicConfig(level=logging.DEBUG)
from azureml.core.model import Model
myws = ws.get(name='mnist1',
                      subscription_id='bcbc4e01-e5d6-42b0-95af-06286341e6ca',	
                      resource_group='mnist3',
       )
print(Model.get_model_path(model_name='sklearn_mnist_model.pkl'))
image_config = ContainerImage.image_configuration(execution_script="score.py", 
                                                  runtime="python", 
                                                  conda_file="myenv.yml")

model = Model.register(workspace = myws,
                       model_path ="./sklearn-mnist/outputs/sklearn_mnist_model.pkl",
                       model_name = "sklearn_mnist",)
aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "MNIST",  "method" : "sklearn"}, 
                                               description='Predict MNIST with sklearn')
# =============================================================================
# global model
#     # retrieve the path to the model file using the model name
# model_path = Model.get_model_path('sklearn_mnist')
# model = joblib.load(model_path)
# 
# =============================================================================
service = Webservice.deploy_from_model(workspace=myws,
                                       name='sklearn-mnist-image',
                                       deployment_config=aciconfig,
                                       models=[model],
                                       image_config=image_config)

service.wait_for_deployment(show_output=True)
print(service.scoring_uri)