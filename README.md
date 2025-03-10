# Multihead-Attention-for-Adaptive-Thresholding-in-Functional-Connectivity-Analysis
We introduce a novel framework for processing and analyzing functional connectivity (FC) matrices derived from functional magnetic resonance imaging (fMRI). The proposed approach integrates multihead attention mechanism and transformer architectures to perform adaptive thresholding of the functional connectivity matrix during training. Unlike conventional proportional thresholding, which is typically applied as a preprocessing step, our method optimizes the selection of relevant connections in an end-to-end manner based on the specific training task.

Graph Neural Networks (GNNs) are commonly used for functional connectivity analysis but are often limited by predefined graph structures and fixed thresholding strategies. In densely connected graphs, GNNs may also suffer from oversmoothing, leading to the loss of discriminative information. Our method bypasses these limitations by employing attention-based mechanisms to dynamically refine connectivity patterns, balancing sparsity and informativeness to enhance functional patterns relevant to the task.

The framework enforces symmetry and sparsity through a tailored loss function, allowing the network to retain the most relevant functional connections while filtering out non-informative ones. This ensures the preservation of essential connectivity patterns.

We validate the method through a benchmarking study on sex classification using the Human Connectome Project (HCP) dataset. Experimental results demonstrate that our framework outperforms GNN-based approaches, achieving higher classification accuracy and more robust feature representations. These findings underscore the effectiveness of attention-based modules in functional connectivity analysis.

![Models](./img/models.png)
