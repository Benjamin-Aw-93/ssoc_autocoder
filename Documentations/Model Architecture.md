### Model Architecture and parameters changes
  * Learning rate reduced from 0.005 to 0.001
  * Language model component seperated from hidden layers
    * Embeddings done seperately before training
    * Embeddings are stored in S3 bucket
    * Embeddings is retrieved from S3 bucket during training
   * Model accuracy improved from 49.9% to >80%
   * Duration for one epoch reduced from 4 hours/epoch to 10 minutes/epoch

## Model Architecture Diagram
<img width="710" alt="image" src="https://github.com/Benjamin-Aw-93/ssoc_autocoder/assets/66168700/4f58f3d2-6841-41d1-8e91-05666d461995">
