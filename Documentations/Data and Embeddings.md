### Naming Convention of train/test data and how to retrieve the embeddings
  * Train and test data CSVs being used are named as ys_train.csv and ys_test.csv respectively
  * df['id'] will be the unique identifier of each row of data
  * Training data id starts from 100000, while Testing data id starts from 0
  * Index of the row is not being used to prevent conflicts (not unique across different CSVs)

<br><br>
**Sample code to retrieve embeddings (title + description)  from S3 bucket**
````
s3 = boto3.client('s3')
bucket_name = 'ag-mom-alphatest'
test =100000 # this refers to the df['id']
file_key = f'ys_embeddings/ys{test}.pt'
response = s3.get_object(Bucket = bucket_name, Key = file_key)
tensor_bytes = response['Body'].read()
tensor = torch.load(BytesIO(tensor_bytes))

tensor
````
**Title-only embeddingst**
````
tensor[0:768]
````
**Description-only embeddings**
````
tensor[769:0]
````

