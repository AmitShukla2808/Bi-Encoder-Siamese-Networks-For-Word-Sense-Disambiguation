# Bi-Encoder-Siamese-Networks-For-Word-Sense-Disambiguation

This repository contains the code for Word-Sense Disambiguation problem using Bi-Encoder Siamese networks. In our work, we have majorly used two models: DistilBert and Tiny-bert for encoders in Siamese Network. Detailed Information about complete work goes below in various sections.

All weight files can be accessed from [here](https://iiithydresearch-my.sharepoint.com/:f:/g/personal/amit_shukla_research_iiit_ac_in/EooiBHSPQsdFhPZkWdh_6m4BAjfNBXxXSdoVHJh9cOfzIg?e=Y6kY1T) All code files, utilies are present in this repo : [Github Repo](https://github.com/AmitShukla2808/Bi-Encoder-Siamese-Networks-For-Word-Sense-Disambiguation)

The complete pipeline is given and discussed below

![INLP_pipeline](https://github.com/user-attachments/assets/d7d5c270-b2e9-466a-860d-a6daf9aa2e22)

## Word Sense Disambiguation

Word Sense Disambiguation (WSD) is a fundamental task in Natural Language Processing (NLP) that involves determining the correct meaning (sense) of a word based on its context. Many words in natural language are polysemous — they have multiple meanings depending on usage. For example, the word "bank" can refer to a financial institution or the side of a river. WSD aims to resolve such ambiguities by assigning the most appropriate sense from a predefined inventory (like WordNet) or through contextual modeling. Accurate WSD is critical for improving machine understanding in applications such as machine translation, information retrieval, question answering, and semantic search. Though there are multiple methods developed in recent past, but all of them have bottlenecks.

Here we bring in out **Siamese Architecture based Bi-encoder network** that aim to resolve this problem in a more efficient manner. Siamese Networks are powerful architectures for tasks involving similarity or comparison between input pairs. In NLP, they are particularly effective for problems like word sense disambiguation, sentence similarity, and paraphrase detection. Their key benefit lies in learning meaningful embeddings that capture semantic relationships. By processing two inputs through identical subnetworks with shared weights, Siamese Networks ensure consistent representation learning and reduce the number of trainable parameters. This architecture excels in low-resource or contrastive learning setups and is naturally suited for tasks requiring matching, ranking, or clustering based on learned distances.


## Bi-Encoder Siamese Networks

In our approach to solve WSD problem, we use Siamese networks using Encoder models like **Tiny-Bert** and **Distil-Bert**. The reason behind choosing these models comes from the fact that they are suited good for downstream task and retain a lot of features from original large **BERT** model. The comparison goes below:

### Model Specifications and Comparison

| Feature               | BERT (Base)         | DistilBERT             | TinyBERT                |
|-----------------------|---------------------|-------------------------|--------------------------|
| Layers (Transformer)  | 12                  | 6                       | 4 / 6 (varies by version) |
| Hidden Size           | 768                 | 768                     | 312 / 768               |
| Attention Heads       | 12                  | 12                      | 12                      |
| Parameters            | ~110M               | ~66M                    | ~14M (TinyBERT-4) / ~66M (TinyBERT-6) |
| Training Objective    | MLM + NSP           | MLM + Knowledge Distil. | MLM + Layer-wise Distil. |
| Speed                 | Baseline            | ~60% faster             | ~7.5× smaller (TinyBERT-4) |
| Accuracy Trade-off    | High accuracy       | ~97% of BERT-base       | ~96–98% of BERT-base    |
| Distillation Type     | N/A                 | Task-agnostic           | Task-specific & general |


The above specifications of the models help us in efficiently training and fine-tuning them for our task even in resource-constrained environment.

## Methodology (An Overview of Our Pipeline)

We have built two variations of **SiamBert** (what we call our model), one using **contrastive loss function** and other **triplet loss function**. These two loss functions perform exceptionally well in disambiguation type of task often used in face detection too.

Our Siamese network is even flexible for using other models as encoder too. These encoders then along with feed forward layers is trained over various datasets for WSD task. As seen above, our model is trained in 2 steps : **Pre-training** and **Fine-tuning**.

### Pre-training of SiamBERT

In this step, based upon the loss function used, we train the model on **3** different datasets derived from **Semcor 3.0**. For contrastive loss, we train both **Tiny SiamBert** and **Distil SiamBert** on **Context-Gloss** and **Context-Hypernymy** datasets. This is very crucial as it helps our model to learn effectively about meanings and categories of various ambigous words used in a large variety of sentences.
For triplet loss, we again train both the models on **Context-Positive-Gloss-Negative-Gloss** dataset. Through this, model tries to learn not only how ambigous words are used in context, it also learns how to differentiate between different meanings of the word.

### Fine-tuning of SiamBERT

In this step, we further fine-tune our model to focus on contextual meaning of the ambigous words instead of just learning up their different meanings. This helps our model to further refine its view of looking at different contexts in which the ambigous words are used and differentiate them effectively. For contrastive loss based model, we used combined **WiC (Words in Context) and Semcor 3.0** data in form of **Sentence 1,  Sentence 2, Ambigous Word,  Label**. Here first three features are self-explanatory while the feature **Label (0/1)** tells if the given word has been used in same context in both sentences or not.
Similarly for triplet loss based model, we used **Semcor Triplets** to fine-tune. In this dataset, we have **Anchor Sentence, Positive Sentence, Negative Sentence**, all containing the ambigous word in target.


## Results and Evaluation
In order to test the effectiveness of our model, we have compared its performance with **GlossBERT** model fine-tuned on **Semcor 3.0**. You can find this model at : [https://huggingface.co/kanishka/GlossBERT](https://huggingface.co/kanishka/GlossBERT) For this task, we have used **SemEval 2013**, **SemEval 2015** and **RAW-C (Related Words in Context)** datasets to assess our performance. The accuracies achieved by all models have been tabulated below:

### SemEval 2015

|            **Model**            |                   **Accuracy**                  |
|---------------------------------|-------------------------------------------------|
| GlossBERT                       |                  57.9439                        |
| Tiny-SiamBERT (contrastive)     |                  **63.3602***                    |
| Distil-SiamBERT (contrastive)   |                  **60.7748***                    |
| Tiny-SiamBERT (triplet)         |                  **68.2242***                    |
| Distil-SiamBERT (triplet)       |                  **74.7663***                    |



### SemEval 2013

|            **Model**            |                   **Accuracy**                  |
|---------------------------------|-------------------------------------------------|
| GlossBERT                       |                  55.3333                        |
| Tiny-SiamBERT (contrastive)     |                  **74.6442***                    |
| Distil-SiamBERT (contrastive)   |                  **75.4150***                    |
| Tiny-SiamBERT (triplet)         |                  **74.6667***                    |
| Distil-SiamBERT (triplet)       |                  **77.3333***                    |



### RAW-C (Related Words in Context)

|            **Model**            |                   **Accuracy**                  |
|---------------------------------|-------------------------------------------------|
| GlossBERT                       |                  61.9047                        |
| Tiny-SiamBERT (contrastive)     |                  **73.9540***                    |
| Distil-SiamBERT (contrastive)   |                  **73.9540***                    |
| Tiny-SiamBERT (triplet)         |                  **62.7976***                    |
| Distil-SiamBERT (triplet)       |                  **69.3452***                    |

Our models consistently beats **GlossBERT** on these 3 evaluation datasets. All other code files, helper function/utilities, datasets and model weights can be found above in the repository.
