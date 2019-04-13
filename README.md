# heart-disease-risk-prediction

# Goal: 

Identify risk factors for heart disease by extracting information from clinical notes using advanced NLP and Deep Learning techniques. 

# Abstract (Project):

Adoption of electronic medical records (EMRs) by healthcare institutions has facilitated the use of analytics for pa-tient care. There is wealth of infor-mation in clinical text, that can be lev-eraged using NLP and Deep Learning techniques. This work proposes a novel way of using stacked embeddings, as a way to improve on work that has al-ready been done in this space as part of i2b2 2014 challenge.

Stacking embeddings, while it is con-ceptually a way to combine multiple embeddings, has shown good results on the i2b2 heart disease risk factors challenge dataset.  By stacking BERT and character embeddings (BERT-CHAR Embedding), we have achieved an F1 score of 93.07% on the test set. This has shown the best results among all other models that we have built as part of this project. This is an encouraging outcome of our experiment towards further research that could surpass the existing benchmark of Deep Learning models built on this shared task (Chokkwijitkul et al.,  2018) which reported an F1 score of 90.81% using BLSTM as well as the most successful system (Roberts et al., 2015) of the i2b2/UTHealth 2014 challenge which reported an F1 score of 92.76%.  Also, our work highlights the fact that use of contextual embeddings further enhances the power of NLP & Deep Learning.  This research work is a step towards an implementation that has the potential to beat the current state-of-the-art results with minimal feature engineering and offer a solution that can perform better than human an-notators.
