import torch
from transformers import AutoModel

class V2pt2(torch.nn.Module):
    def __init__(self, 
        lm_filepath: "model/distilbert",
        max_level: 5
        ):
        # Initialise the class, not sure exactly what this does
        # Ben: Should be similar to super()?
        super().__init__()

        # Load the DistilBert model which will generate the embeddings
        self.l1 = AutoModel.from_pretrained(lm_filepath)
        for param in self.l1.parameters():
            param.requires_grad = False
        self.max_level = max_level
        # Generate counts of each digit SSOCs
        SSOC_1D_count = 9 #len(self.encoding['SSOC_1D']['ssoc_idx'].keys())
        SSOC_2D_count = 42 #len(self.encoding['SSOC_2D']['ssoc_idx'].keys())
        SSOC_3D_count = 144 #len(self.encoding['SSOC_3D']['ssoc_idx'].keys())
        SSOC_4D_count = 413 #len(self.encoding['SSOC_4D']['ssoc_idx'].keys())
        SSOC_5D_count = 997 #len(self.encoding['SSOC_5D']['ssoc_idx'].keys())
        # Stack 1: Predicting 1D SSOC (9)
        if max_level >= 1:
            self.ssoc_1d_stack = torch.nn.Sequential(
                torch.nn.Linear(1536, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, SSOC_1D_count)
            )
        # Stack 2: Predicting 2D SSOC (42)
        if max_level >= 2:
            # Adding the predictions from Stack 1 to the word embeddings
            # n_dims_2d = 1545
            n_dims_2d = 1536 + SSOC_1D_count
            self.ssoc_2d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_2d, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, SSOC_2D_count)
            )
        # Stack 3: Predicting 3D SSOC (144)
        if max_level >= 3:
            # Adding the predictions from Stacks 1 and 2 to the word embeddings
            # n_dims_3d = 1587
            n_dims_3d = 1536 + SSOC_1D_count + SSOC_2D_count
            self.ssoc_3d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_3d, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, SSOC_3D_count)
            )
        # Stack 4: Predicting 4D SSOC (413)
        if max_level >= 4:
            # Adding the predictions from Stacks 1, 2, and 3 to the word embeddings
            # n_dims_4d = 1731
            n_dims_4d = 1536 + SSOC_1D_count + SSOC_2D_count + SSOC_3D_count
            self.ssoc_4d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_4d, 768),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(768, SSOC_4D_count)
            )
        # Stack 5: Predicting 5D SSOC (997)
        if max_level >= 5:
            # Adding the predictions from Stacks 1, 2, 3 and 4 to the word embeddings
            # n_dims_5d = 2144
            n_dims_5d = 1536 + SSOC_1D_count + SSOC_2D_count + SSOC_3D_count + SSOC_4D_count
            self.ssoc_5d_stack = torch.nn.Sequential(
                torch.nn.Linear(n_dims_5d, 1024),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(1024, SSOC_5D_count)
            )

    def forward(self, title_ids, title_mask, text_ids, text_mask):
        # Obtain the sentence embeddings from the DistilBERT model
        # Do this for both the job title and description text
        title_embeddings = self.l1(input_ids = title_ids, attention_mask = title_mask)
        title_hidden_state = title_embeddings[0]
        title_vec = title_hidden_state[:, 0]
        text_embeddings = self.l1(input_ids = text_ids, attention_mask = text_mask)
        text_hidden_state = text_embeddings[0]
        text_vec = text_hidden_state[:, 0]
        # Concatenate both vectors together
        X = torch.cat((title_vec, text_vec), dim = 1)
        # Initialise a dictionary to hold all the predictions
        predictions = {}
        # 1D Prediction
        if self.max_level >= 1:
            predictions['SSOC_1D'] = self.ssoc_1d_stack(X)
        # 2D Prediction
        if self.max_level >= 2:
            X = torch.cat((X, predictions['SSOC_1D']), dim=1)
            predictions['SSOC_2D'] = self.ssoc_2d_stack(X)
        # 3D Prediction
        if self.max_level >= 3:
            X = torch.cat((X, predictions['SSOC_2D']), dim=1)
            predictions['SSOC_3D'] = self.ssoc_3d_stack(X)
        # 4D Prediction
        if self.max_level >= 4:
            X = torch.cat((X, predictions['SSOC_3D']), dim=1)
            predictions['SSOC_4D'] = self.ssoc_4d_stack(X)
        # 5D Prediction
        if self.max_level >= 5:
            X = torch.cat((X, predictions['SSOC_4D']), dim=1)
            predictions['SSOC_5D'] = self.ssoc_5d_stack(X)
        return {f'SSOC_{i}D': predictions[f'SSOC_{i}D'] for i in range(1, self.max_level + 1)}
