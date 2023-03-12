
Implemented the model described in the ”Coherent Hierarchical Multi-Label Classification Networks” paper for a custom dataset using
PyTorch and scikit-learn, achieving 97% Mean Average Precision.



Training and validation accuracy for 10 epoch: 

(The accuracy is calculated accepting the output as true if and only if the entire hierarchical path was predicted correctly)

![accuracy](https://user-images.githubusercontent.com/58374690/224551083-b4f85a34-134e-4599-ab27-5d4fbc990220.png)



Training and validation loss for 10 epoch:

(With binary cross-entropy with logits loss. For torch.nn.BCEWithLogitsLoss, more at https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

![loss](https://user-images.githubusercontent.com/58374690/224551088-1762c06f-d46e-45e3-be16-a55074e1cb16.png)



Training and validation average precision score for 10 epoch:

(For Mean Average Precision, more at https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

![score](https://user-images.githubusercontent.com/58374690/224551090-c5bb4a3f-c8b4-40e6-a9a7-65ea7d08337e.png)
