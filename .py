import kagglehub

# Download latest version
path = kagglehub.dataset_download("enider/yawdd-dataset", path='./data/')

print("Path to dataset files:", path)