# reuse registry environment image
FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.1-cuda12.1

# Customize environment by adding packages
RUN pip install acvl-utils==0.2 dynamic-network-architectures==0.3.1 batchgenerators==0.25 numpy==1.24 scikit-image==0.19.3 SimpleITK==2.2.1 nnunetv2
RUN pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
