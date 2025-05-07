# Bi-Encoder-Siamese-Networks-For-Word-Sense-Disambiguation

This repository contains the code for Word-Sense Disambiguation problem using Bi-Encoder Siamese networks. In our work, we have majorly used two models: DistilBert and Tiny-bert for encoders in Siamese Network. Detailed Information about complete work goes below in various sections.

All weight files can be accessed from [here](https://iiithydresearch-my.sharepoint.com/:f:/g/personal/amit_shukla_research_iiit_ac_in/EooiBHSPQsdFhPZkWdh_6m4BAjfNBXxXSdoVHJh9cOfzIg?e=Y6kY1T)

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

