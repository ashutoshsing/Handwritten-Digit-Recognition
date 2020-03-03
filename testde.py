import json
import numpy as np
from utils import load_data
from azureml.core.webservice import Webservice
from azureml.core import Workspace as ws
#from azureml.core.model import Model
#from azureml.core.webservice import AciWebservice
import os
#import matplotlib
import matplotlib.pyplot as plt
# find 30 random samples from test set
n = 30
myws = ws.get(name='mnist1',
                      subscription_id='bcbc4e01-e5d6-42b0-95af-06286341e6ca',	
                      resource_group='mnist3',
       )
#model = Model(myws, 'sklearn-mnist')
#web = 'https://mlworkspace.azure.ai/portal/subscriptions/bcbc4e01-e5d6-42b0-95af-06286341e6ca/resourceGroups/mnist3/providers/Microsoft.MachineLearningServices/workspaces/mnist1/deployments/mnist'
#print(Webservice.list(myws)[0].name)
service = Webservice(myws , 'mnist')
#print(type(model))
data_folder = os.path.join(os.getcwd(), 'data')
X_test = load_data(os.path.join(data_folder, 'test-images.gz'), False) / 255.0
y_test = load_data(os.path.join(data_folder, 'test-labels.gz'), True).reshape(-1)
sample_indices = np.random.permutation(X_test.shape[0])[0:n]
 
test_samples = json.dumps({"data": X_test[sample_indices].tolist()})
test_samples = bytes(test_samples, encoding='utf8')
#result = service.run(input_data=test_samples) 
 # predict using the deployed model
result = service.run(input_data=test_samples)
 
 # compare actual value vs. the predicted values:
i = 0
plt.figure(figsize = (20, 1))
 
for s in sample_indices:
    plt.subplot(1, n, i + 1)
    plt.axhline('')
    plt.axvline('')
     
     # use different color for misclassified sample
    font_color = 'red' if y_test[s] != result[i] else 'black'
    clr_map = plt.cm.gray if y_test[s] != result[i] else plt.cm.Greys
     
    plt.text(x=10, y =-10, s=result[i], fontsize=18, color=font_color)
    plt.imshow(X_test[s].reshape(28, 28), cmap=clr_map)
     
    i = i + 1
plt.show()


# =============================================================================
# service = 'https://mlworkspace.azure.ai/portal/subscriptions/bcbc4e01-e5d6-42b0-95af-06286341e6ca/resourceGroups/mnist3/providers/Microsoft.MachineLearningServices/workspaces/mnist1/deployments/mnist'

# =============================================================================
