# Seismic First-Break Picking with U-Net
This project implements a deep learning-based seismic first-break picking
system using a U-Net architecture with a ResNet encoder. The model is deployed
as a web application using Flask.

## Features
- Upload seismic data in `.npz` format
- Automatic first-break prediction
- Visualization of prediction results
- Downloadable prediction outputs

## Dataset Attribution
This project uses seismic data derived from an open-source GitHub repository
licensed under the Apache License, Version 2.0.

The dataset was used solely for training and evaluation purposes.
All preprocessing pipelines, model architectures, and deployment
implementations were developed independently.

## Deployment
- Backend: Flask + TensorFlow
- Platform: Render (Cloud)
