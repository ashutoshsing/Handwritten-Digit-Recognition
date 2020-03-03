from azureml.train.sklearn import SKLearn
from azureml.core import Workspace as ws 
ds = ws.get_default_datastore()
script_params = {
    '--data-folder': ds.path('mnist').as_mount(),
    '--regularization': 0.5
}

est = SKLearn(source_directory=script_folder,
                script_params=script_params,
                compute_target=compute_target,
                entry_script='train.py')

run = exp.submit(config=est)
run