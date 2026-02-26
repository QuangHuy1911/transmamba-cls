\documentclass[runningheads]{llncs}

\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}   
\usepackage{float}
\usepackage{placeins}
\usepackage{needspace}
\usepackage{subcaption}
\begin{document}

\title{Enhancing Classic Classifiers with SLM-Derived Semantic Features for DDoS Detection}

\titlerunning{SLM-Enhanced Logistic Regression for DDoS Detection}

\author{
\author{
Long Truong\inst{1}\orcidID{0009-0003-6935-6697}\and Khoi Nguyen\inst{1}\orcidID{0009-0007-0928-2015} 
 \and
Tai Nguyen\inst{1}\orcidID{0009-0006-1463-2940}
}  
}

\authorrunning{Long Truong et al.}

\institute{
Department of Computer Science, FPT University, Vietnam\\
\email{\{khoin, longt5, tain\}@fpt.edu.vn}
}

\maketitle

\begin{abstract}
Distributed Denial-of-Service (DDoS) attacks remain a critical threat to networked systems, requiring detection frameworks that balance effectiveness and deployability. While lightweight classifiers such as Logistic Regression (LR) offer low computational overhead, they often underperform when relying solely on conventional tabular features, whereas deep learning approaches achieve high accuracy at substantial computational cost.

To address this trade-off, we propose Embedding-Driven Attention-based Detection (EDAD), a hybrid DDoS detection framework that integrates PCA-reduced statistical flow features with semantic embeddings extracted from a Small Language Model (SLM). An attention-based fusion mechanism adaptively combines heterogeneous representations, and LR is used for final classification to preserve lightweight inference.

Experiments on CICDDoS2019 show that the proposed method improves accuracy from 86\% (LR+PCA baseline) to 97.04\%, increasing F1-score from 0.87 to 0.9355. On AdDDoSDN, the model achieves an F1-score of 0.9674, demonstrating strong cross-environment generalization. These results indicate that enriching classical feature pipelines with SLM-derived semantic representations enhances detection performance while preserving computational efficiency. 

\keywords{DDoS Detection \and Small Language Models \and Logistic Regression \and Feature Fusion \and Network Security}
\end{abstract}

\section{Introduction}

Distributed Denial-of-Service (DDoS) attacks continue to threaten the availability of networked systems and online services, motivating extensive research on intelligent intrusion detection systems (IDS). Machine learning-based approaches have become central to automated DDoS detection due to their ability to learn discriminative patterns from large-scale traffic data \cite{bala2024}.

Traditional supervised models such as Logistic Regression (LR), SVM, Random Forest, and KNN remain widely adopted because of their low computational cost and ease of deployment \cite{abiramasundari2025,ismail2022}. However, their effectiveness heavily depends on feature representation quality. Even with dimensionality reduction techniques such as PCA, linear classifiers often struggle to capture the complex and heterogeneous characteristics of modern DDoS traffic. For example, the EDAD framework reports that LR achieves only approximately 86\% accuracy on CICDDoS2019 \cite{abiramasundari2025}.

Deep learning models, including CNN-, RNN-, and Transformer-based architectures, significantly improve detection performance by learning hierarchical and contextual representations \cite{akgun2022,ddosbert2025}. In particular, DDoSBERT achieves approximately 99.5\% accuracy on CICDDoS2019 \cite{ddosbert2025}. However, such end-to-end deep models typically require substantial computational resources and GPU acceleration, limiting their suitability for real-time and resource-constrained deployment.

These observations highlight a trade-off between detection accuracy and computational efficiency. While classical models are lightweight but representation-limited, deep architectures provide powerful feature learning at high computational cost. Hybrid approaches that decouple representation learning from lightweight classification remain relatively underexplored in DDoS detection.

Motivated by this gap, we propose a hybrid framework that integrates PCA-reduced statistical flow features with SLM-derived semantic embeddings through an attention-based fusion mechanism, while retaining Logistic Regression for efficient inference. On CICDDoS2019, the proposed method improves accuracy from 86\% to 97.04\% and increases F1-score from 0.87 to 0.9355 compared to the PCA-based LR baseline. Additional evaluation on AdDDoSDN achieves an F1-score of 0.9674, demonstrating strong cross-environment generalization.

\section{Related Work}

\subsection{Machine Learning and Deep Learning for DDoS Detection}

Classical supervised machine learning methods have long been applied to DDoS detection due to their simplicity and deployment efficiency. Common classifiers include Logistic Regression (LR), Support Vector Machines (SVM), Random Forest (RF), and k-Nearest Neighbors (KNN), typically trained on flow-level statistical features \cite{abiramasundari2025,ismail2022}. However, such approaches rely heavily on handcrafted or PCA-transformed features, which may limit their ability to model highly nonlinear traffic patterns.

Deep learning techniques have been introduced to improve representational capacity. CNN- and RNN-based models learn hierarchical and temporal features directly from traffic data \cite{akgun2022,ismail2022}. More recently, Transformer-based architectures leverage self-attention mechanisms to capture contextual dependencies and achieve state-of-the-art performance in DDoS detection \cite{ddosbert2025}. Despite their effectiveness, these models generally involve substantial computational cost and hardware requirements \cite{bala2024}.

\subsection{Representation-Centric and Hybrid Approaches}

An alternative research direction focuses on separating representation learning from classification. Pretrained language models such as BERT demonstrate that contextual embeddings can serve as transferable feature extractors for downstream tasks without full end-to-end fine-tuning \cite{devlin2019}. Representation-centric approaches further show that integrating learned embeddings with classical classifiers can enhance performance while preserving efficiency.

In DDoS detection, existing studies predominantly adopt either lightweight classifiers on tabular features \cite{abiramasundari2025} or fully end-to-end deep architectures \cite{ddosbert2025}. Hybrid frameworks that employ Small Language Models (SLMs) specifically for semantic feature extraction—combined with adaptive fusion and lightweight classification—remain relatively underexplored.

\subsection{Positioning of This Work}

Compared with prior studies, our framework:

\begin{itemize}
    \item Employs an SLM solely as a semantic feature extractor rather than as an end-to-end classifier;
    \item Integrates SLM-derived embeddings with PCA-reduced tabular features through an attention-based fusion mechanism;
    \item Retains Logistic Regression for lightweight inference and deployment efficiency;
    \item Explicitly targets the accuracy--deployability balance in practical DDoS detection scenarios.
\end{itemize}

\section{Methodology}

\subsection{Overview}

The overall architecture of EDAD is illustrated in Figure~\ref{fig:pipeline}. 
The framework follows a dual-branch design: (i) classical tabular feature processing and (ii) semantic embedding extraction using a Small Language Model (SLM). 
These representations are integrated through an attention-based fusion mechanism and classified using Logistic Regression.

\begin{figure}[t]
    \centering
    \includegraphics[width=0.8\linewidth]{final.drawio.png}
    \caption{Pipeline of the proposed EDAD framework. The left branch processes standardized and PCA-reduced tabular features, while the right branch extracts semantic embeddings using DDoSBert. An attention-based fusion module adaptively integrates heterogeneous representations before Logistic Regression classification.}
    \label{fig:pipeline}
\end{figure}

The pipeline consists of the following stages: data cleaning, stratified train–test split, feature standardization, SMOTE (training only), PCA reduction, prompt construction, DDoSBert encoding, attention-based fusion, and final classification.

\subsection{Tabular Feature Processing}

To prevent data leakage, we adopt a split-first protocol with a stratified 70/30 train–test ratio. 
Numerical features are standardized using z-score normalization. 
Class imbalance is addressed using SMOTE oversampling applied exclusively to the training set \cite{chawla2002smote}. 
Dimensionality reduction is performed using Principal Component Analysis (PCA) with 95\% variance retention \cite{jolliffe2002pca}.

Let $z_i \in \mathbb{R}^{k}$ denote the resulting PCA-reduced tabular representation for flow $i$.

\subsection{Semantic Embedding Extraction}

Selected flow attributes (Flow Duration, Flow Bytes/s, Flow Packets/s) are converted into structured natural-language prompts describing traffic behavior. 
After tokenization (maximum length 128), DDoSBert produces contextual hidden states $H_i \in \mathbb{R}^{T \times 768}$.

A fixed-length semantic representation is obtained using mean pooling:

\[
e_i = \frac{1}{T}\sum_{t=1}^{T} h_{it}
\]

Mean pooling provides a stable global representation without additional trainable parameters \cite{reimers2019sentence}.

\subsection{Attention-Based Feature Fusion}

Let $z_i$ denote PCA-reduced tabular features and $e_i$ denote semantic embeddings. 
The two representations are concatenated:

\[
c_i = [z_i; e_i]
\]

We compute an adaptive weight using a lightweight MLP-based gating mechanism inspired by additive attention and feature recalibration mechanisms \cite{bahdanau2015neural,hu2018squeeze}:

\[
w_i = \sigma(W_2 \tanh(W_1 c_i))
\]

The fused representation is defined as:

\[
f_i = [w_i z_i; (1 - w_i)e_i]
\]

This design enables adaptive weighting between statistical and semantic feature spaces while maintaining computational efficiency.

\subsection{Classification}

The fused representation $f_i$ is classified using Logistic Regression trained with binary cross-entropy loss.
\FloatBarrier
\section{Results}

\subsection{Experimental Setup}

We evaluate the proposed hybrid framework on two publicly available
benchmark datasets: CICDDoS2019~\cite{cicddos2019} and AdDDoSDN~\cite{mokti2025}. 
The hybrid model is compared against a classical baseline consisting 
of Logistic Regression (LR) combined with Principal Component Analysis (PCA), 
as commonly adopted in prior DDoS detection studies~\cite{abiramasundari2025,ismail2022}.

Evaluation metrics include Accuracy, Precision, Recall, and F1-score.
A stratified 70--30 train--test split is used. To assess robustness and
generalization stability, 5-fold cross-validation is conducted,
which is widely adopted in machine learning-based intrusion detection~\cite{ismail2022}.

\subsection{Results on CICDDoS2019}
The proposed hybrid framework improves accuracy by more than 
11 percentage points compared to the PCA-based baseline. 
The increase in F1-score demonstrates a better balance between 
precision and recall, indicating enhanced discriminative capability.
\begin{table}[!ht]
\centering
\caption{Performance comparison on CICDDoS2019}
\begin{tabular}{lcccc}
\toprule
Model & Accuracy & Precision & Recall & F1-score \\
\midrule
LR + PCA (Baseline) & 0.86 & 0.84 & 0.90 & 0.87 \\
Hybrid (Proposed) & 0.9704 & 0.9273 & 0.9439 & 0.9355 \\
\bottomrule
\end{tabular}
\end{table}



\subsection{Results on AdDDoSDN}    
Despite the different traffic characteristics in SDN environments,
the hybrid model maintains strong detection capability, achieving
an F1-score of 0.9674. This result indicates good cross-environment
generalization.

\begin{table}[t]
\centering
\caption{Hybrid model performance on AdDDoSDN}
\begin{tabular}{lc}
\toprule
Metric & Score \\
\midrule
Accuracy & 0.9462 \\
Precision & 0.9799 \\
Recall & 0.9552 \\
F1-score & 0.9674 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Cross-Validation Stability}

The low standard deviations across folds indicate stable performance
and confirm that the improvement is not dependent on a specific
train--test split.
\begin{table}[!ht]
\centering
\caption{5-fold cross-validation results}
\begin{tabular}{lcc}
\toprule
Dataset & Mean F1-score & Std \\
\midrule
CICDDoS2019 & 0.9247 & 0.0078 \\
AdDDoSDN & 0.9721 & 0.0023 \\
\bottomrule
\end{tabular}
\end{table}


\subsection{Confusion Matrix Analysis}

To further examine class-wise prediction behavior, we analyze the
confusion matrix on the AdDDoSDN dataset.

\begin{table}[!ht]
\centering
\caption{Confusion matrix summary on AdDDoSDN}
\label{tab:conf_add}
\begin{tabular}{lcc}
\toprule
 & Predicted Negative & Predicted Positive \\
\midrule
True Negative  & 11,937 & 1,323 \\
True Positive  & 3,019  & 64,388 \\
\bottomrule
\end{tabular}
\end{table}

The results indicate that the proposed hybrid model correctly
identifies the majority of attack samples (TP = 64,388) while
maintaining a relatively low number of false positives (FP = 1,323).
The low false negative rate further confirms the robustness of the
semantic-enhanced representation in distinguishing malicious traffic
from benign flows.

\FloatBarrier


\subsection{t-SNE Visualization}

To qualitatively examine the geometric structure of the learned
feature space, we apply t-distributed Stochastic Neighbor
Embedding (t-SNE) to visualize both the baseline and the
proposed hybrid representations.

\begin{figure}[!ht]
\centering

\begin{subfigure}[t]{0.67\linewidth}
\centering
\includegraphics[width=\linewidth]{2019.png}
\caption{CICDDoS2019}
\end{subfigure}
\hfill
\begin{subfigure}[t]{0.67\linewidth}
\centering
\includegraphics[width=\linewidth]{AdDDoSDN.png}
\caption{AdDDoSDN}
\end{subfigure}

\caption{t-SNE visualization of feature representations on two datasets.
Within each dataset, the left subplot corresponds to the baseline
PCA-reduced tabular features, while the right subplot corresponds
to the proposed semantic-enriched hybrid representation.}
\label{fig:tsne}
\end{figure}

As shown in Fig.~\ref{fig:tsne}, the baseline tabular features produce
dispersed clusters with noticeable inter-class overlap.
In contrast, the proposed hybrid representation yields
more compact intra-class structures and clearer class
separation across both datasets.
These geometric improvements are consistent with the
observed gains in F1-score, suggesting that the
semantic-enriched fusion produces more discriminative
traffic representations and thereby improves the
effectiveness of Logistic Regression.



\section{Conclusion}

In this work, we proposed EDAD, a hybrid DDoS detection framework that enhances lightweight Logistic Regression with SLM-derived semantic embeddings. By decoupling representation learning from classification and integrating PCA-reduced statistical features with semantic embeddings through an attention-based fusion mechanism, the framework achieves a better balance between detection performance and deployment efficiency.

Experimental results show that the proposed method improves accuracy on CICDDoS2019 from 86\% (LR + PCA baseline) to 97.04\%, with consistent gains in F1-score. The model also demonstrates strong cross-environment generalization on AdDDoSDN, achieving an F1-score of 0.9674. Cross-validation results further confirm the robustness and stability of the approach.

Overall, this study demonstrates that semantic enrichment can substantially strengthen classical linear classifiers without relying on end-to-end deep architectures. Future work will focus on computational cost analysis, comprehensive ablation studies, and further optimization of the fusion mechanism to improve efficiency in real-world deployment scenarios.





\bibliographystyle{splncs04}
\bibliography{references}

\end{document}


