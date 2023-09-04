### Results of experiments done between July 2023 to August 2023
  * Using ChatGPT-3 generated job ads 
  * Tested using XGBoost
  * Ran model training with more epochs

|          Model         	|   Details  	|   Epochs   	|     1D ACC    	|     2D ACC    	|     3D ACC    	|     4D ACC    	|     5D ACC    	|     Top-3     	|     Top-5     	|
|:----------------------:	|:----------:	|:----------:	|:-------------:	|:-------------:	|:-------------:	|:-------------:	|:-------------:	|:-------------:	|:-------------:	|
| Hierachical Classifier 	|  W/O GenAI 	|      35    	|     92.70%    	|     89.79%    	|     87.13%    	|     84.29%    	|     79.38%    	|     91.52%    	|     94.39%    	|
| Hierachical Classifier 	|  W/O GenAI 	|     100    	|     94.01%    	|     91.90%    	|     89.55%    	|     87.23%    	|     84.01%    	|     94.01%    	|     96.47%    	|
| Hierachical Classifier 	| With GenAI 	|     100    	|     93.91%    	|     92.15%    	|     90.42%    	|     88.44%    	|     84.50%    	|     93.70%    	|     96.23%    	|
|  XGBoost max_depth = 3 	|  W/O GenAI 	|      -     	|     89.03%    	|     85.88%    	|     83.46%    	|     82.04%    	|     79.69%    	|     91.38%    	|     93.88%    	|
| XGBoost max_depth = 60 	|  W/O GenAI 	|      -     	|     90.93%    	|     88.62%    	|     86.51%    	|     84.57%    	|     82.32%    	|     92.49%    	|     94.78%    	|


  * Tested different language models


|     Model     	| Accuracy 	|
|:-------------:	|:--------:	|
| RoBERTa Large 	|    50%   	|
|   DistilBERT  	|  82.31%  	|
