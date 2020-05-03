# DocumentSummarization

Summarization is a classic text processing problem. Broadly speaking, given one or more documents, the goal is to obtain a concise piece
of text that contains the most important information of the document. Summarization has been studied for the past years in various
settings—multi-document summarization, summarization using concept spaces in machine learning, Determinantal point processes, submodular optimization. Each domain throws up its own set of improvement, which can be combined together.
\bigbreak
While there have been many approaches to automatic summarization, our work
is directly inspired by the framework of Lin and Bilmes (2011) \cite{}.  In this framework, each of the constraints (relevance, redundancy, etc.) is captured as a submodular function and the objective is to maximize their sum. A simple greedy algorithm is guaranteed to produce an approximately optimal summary. This framework obtains the best results on the DUC 2004 dataset. In this this project, we aim at comparing this approach with a more classical one (MMR) using an accelerated modified greedy algorithm on the Opinosis dataset.


## Problem formulation
We introduce first the following notations :
-  The ground set $V$ corresponds to all the sentences in a document.
- Extractive document summarization: select a small subset $S$ that accurately represents the document (ground set V).
- The summary is required to be length-limited : \\
        - c(S) : cost  for sentences S (e.g., the number of words in all sentences of S) \\
        - b: the budget (e.g., the largest length allowed) \\
- A set function $f : 2^V \xrightarrow{} \mathbf{R}$ measures the quality of the summary S,

Thus, we can formulate the problem as follow
$\begin{definition}{Document Summarization Optimization Problem}
        \begin{equation}
           S^* \in \argmax_{S\subset V} f(S) \qquad \text{subject to} \quad c(S) \leq B
        \end{equation}
\end{definition}$

This problem is NP-hard and we are looking for some methods to compute near-optimal strategies.
## Submodular optimization
    ### Definition
        Let $V$ be a finite set, and denote by $2^V$ the power set of $V$, i.e., the family of all subsets of $V$. A function $f: 2^V \xrightarrow{} \mathbf{R}$ is called submodular if, for each $A,B \in 2^V$, we have: 
        \begin{equation}
            f(A) + f(B) \geq f(A\cap B) +f(A \cup B).
        \end{equation}

## What you can find in this repo
We tested different submodular functions and try the greedy and double greedy algorithm to sum up texts. Please read the paper, the report, or presentations (slides) for more informations.

**Copying my code for your assignment is an academic offence.**

`A Class of Submodular Functions for Document Summarization.pdf` ---- The article we are interested in.

`OpinosisDataset1.0_0` ---- The dataset we are using.

`Submod.py` ---- the class we designed for experiment.

`Experiments_double_nobudget.ipynb` ---- jupyter notebook which contains the experiments on all data.

`Experiments_lazy_double_nobudget.ipynb` ---- jupyter notebook which contains the experiments on most data. Some large data are excluded considering the running time.
