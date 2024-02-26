**Transformers: Attention is all you need**

"Attention is All You Need" is a research paper published in 2017 by Google researchers, which introduced the Transformer model, a novel architecture that revolutionized the field of natural language processing (NLP) and became the basis for the LLMs we now know - such as GPT, PaLM and others. The paper proposes a neural network architecture that replaces traditional recurrent neural networks (RNNs) and convolutional neural networks (CNNs) with an entirely attention-based mechanism.

The Transformer model uses self-attention to compute representations of input sequences, which allows it to capture long-term dependencies and parallelize computation effectively. The authors demonstrate that their model achieves state-of-the-art performance on several machine translation tasks and outperforms previous models that rely on RNNs or CNNs.

The Transformer architecture consists of an encoder and a decoder, each of which is composed of several layers. Each layer consists of two sub-layers: a multi-head self-attention mechanism and a feed-forward neural network. The multi-head self-attention mechanism allows the model to attend to different parts of the input sequence, while the feed-forward network applies a point-wise fully connected layer to each position separately and identically.

The Transformer model also uses residual connections and layer normalization to facilitate training and prevent overfitting. In addition, the authors introduce a positional encoding scheme that encodes the position of each token in the input sequence, enabling the model to capture the order of the sequence without the need for recurrent or convolutional operations.

You can read the Transformers paper [here](https://arxiv.org/abs/1706.03762).

  

**Domain-specific training: BloombergGPT**

  

  

  

BloombergGPT, developed by Bloomberg, is a large Decoder-only language model. It underwent pre-training using an extensive financial dataset comprising news articles, reports, and market data, to increase its understanding of finance and enabling it to generate finance-related natural language text. The datasets are shown in the image above.

During the training of BloombergGPT, the authors used the Chinchilla Scaling Laws to guide the number of parameters in the model and the volume of training data, measured in tokens. The recommendations of Chinchilla are represented by the lines Chinchilla-1, Chinchilla-2 and Chinchilla-3 in the image, and we can see that BloombergGPT is close to it.

While the recommended configuration for the team’s available training compute budget was 50 billion parameters and 1.4 trillion tokens, acquiring 1.4 trillion tokens of training data in the finance domain proved challenging. Consequently, they constructed a dataset containing just 700 billion tokens, less than the compute-optimal value. Furthermore, due to early stopping, the training process terminated after processing 569 billion tokens.

The BloombergGPT project is a good illustration of pre-training a model for increased domain-specificity, and the challenges that may force trade-offs against compute-optimal model and training configurations.

You can read the BloombergGPT article [here](https://arxiv.org/abs/2303.17564).

  

**Week 1 resources**

  

Below you'll find links to the research papers discussed in this weeks videos. You don't need to understand all the technical details discussed in these papers - **you have already seen the most important points you'll need to answer the quizzes** in the lecture videos.

However, if you'd like to take a closer look at the original research, you can read the papers and articles via the links below.

  

**Transformer Architecture**

-   [**Attention is All You Need**](https://arxiv.org/pdf/1706.03762)

- This paper introduced the Transformer architecture, with the core “self-attention” mechanism. This article was the foundation for LLMs.

[**BLOOM: BigScience 176B Model**](https://arxiv.org/abs/2211.05100)

- BLOOM is a open-source LLM with 176B parameters (similar to GPT-4) trained in an open and transparent way. In this paper, the authors present a detailed discussion of the dataset and process used to train the model. You can also see a high-level overview of the model [here](https://bigscience.notion.site/BLOOM-BigScience-176B-Model-ad073ca07cdf479398d5f95d88e218c4)

.

[**Vector Space Models**](https://www.coursera.org/learn/classification-vector-spaces-in-nlp/home/week/3)

-   - Series of lessons from DeepLearning.AI's Natural Language Processing specialization discussing the basics of vector space models and their use in language modeling.

**Pre-training and scaling laws**

-   [**Scaling Laws for Neural Language Models**](https://arxiv.org/abs/2001.08361)
-   - empirical study by researchers at OpenAI exploring the scaling laws for large language models.

**Model architectures and pre-training objectives**

-   [**What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?**](https://arxiv.org/pdf/2204.05832.pdf)

- The paper examines modeling choices in large pre-trained language models and identifies the optimal approach for zero-shot generalization.

[**HuggingFace Tasks**](https://huggingface.co/tasks)

**and** [**Model Hub**](https://huggingface.co/models)

- Collection of resources to tackle varying machine learning tasks using the HuggingFace library.

[**LLaMA: Open and Efficient Foundation Language Models**](https://arxiv.org/pdf/2302.13971.pdf)

-   - Article from Meta AI proposing Efficient LLMs (their model with 13B parameters outperform GPT3 with 175B parameters on most benchmarks)

**Scaling laws and compute-optimal models**

-   [**Language Models are Few-Shot Learners**](https://arxiv.org/pdf/2005.14165.pdf)

-  This paper investigates the potential of few-shot learning in Large Language Models.

[**Training Compute-Optimal Large Language Models**](https://arxiv.org/pdf/2203.15556.pdf)

- Study from DeepMind to evaluate the optimal model size and number of tokens for training LLMs. Also known as “Chinchilla Paper”.

[**BloombergGPT: A Large Language Model for Finance**](https://arxiv.org/pdf/2303.17564.pdf)

- LLM trained specifically for the finance domain, a good example that tried to follow chinchilla laws.
