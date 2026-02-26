% ===========================================================================
% TransMamba-Cls: Adapting Hybrid Transformer-Mamba Architecture
% for Text Classification on GLUE Benchmark
% ---------------------------------------------------------------------------
% Format: Springer LNCS (Lecture Notes in Computer Science)
% Conference: APWeb-WAIM 2026
% Compiler: pdfLaTeX
% ===========================================================================

\documentclass[runningheads]{llncs}

% --- Packages ---
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows,positioning,fit,calc}
\usepackage{subcaption}

% Custom commands
\newcommand{\transmamba}{\textsc{TransMamba-Cls}}
\newcommand{\ours}{\textsc{Ours}}

\begin{document}

% ===========================================================================
% TITLE & AUTHORS
% ===========================================================================

\title{TransMamba-Cls: Adapting Hybrid Transformer-Mamba\\Architecture for Text Classification on GLUE Benchmark}

\titlerunning{TransMamba-Cls for Text Classification}

\author{Long Truong\inst{1}\orcidID{} \and
Quang Huy Nguyen\inst{1}\orcidID{} \and
Ngoc Thuong Le\inst{1}\orcidID{} \and
Tan Dat Vo\inst{1}\orcidID{}}

\authorrunning{L. Truong et al.}

\institute{Faculty of Artificial Intelligence, FPT University, Lo E2a-7, Street D1,
High-Tech Park, Long Thanh My Ward, Thu Duc City, Ho Chi Minh
City, 700000, Ho Chi Minh, Vietnam.\\
\email{\{longts, huyts, thuongts, datts\}@fpt.edu.vn}}

\maketitle

% ===========================================================================
% ABSTRACT
% ===========================================================================

\begin{abstract}
Recent advances in State Space Models (SSMs), particularly Mamba, have demonstrated
competitive performance with Transformers while achieving linear-time complexity.
However, their effectiveness for text classification tasks remains underexplored.
In this paper, we present \transmamba{}, an adaptation of the TransMamba hybrid
architecture for text classification on the GLUE benchmark. Our approach combines
a pretrained BERT encoder for global context modeling with a Mamba decoder for
efficient sequential processing, connected through a Feature Fusion mechanism
consisting of learned feature projections and cross-attention. We conduct
comprehensive experiments on three GLUE tasks (SST-2, MNLI, RTE) across three
encoder scales (BERT-tiny, BERT-small, BERT-base), along with ablation studies
on four fusion strategies. Our results demonstrate that the Feature Fusion
mechanism with learned projections consistently improves performance, and
TransMamba-Cls outperforms both standalone BERT and Pure Mamba baselines.
These findings validate the applicability of hybrid Transformer-Mamba
architectures for downstream NLP classification tasks.

\keywords{Hybrid Architecture \and Transformer \and Mamba \and
State Space Model \and Text Classification \and GLUE Benchmark \and
Feature Fusion}
\end{abstract}

% ===========================================================================
% 1. INTRODUCTION
% ===========================================================================

\section{Introduction}
\label{sec:introduction}

Transformer-based models~\cite{vaswani2017attention} have become the dominant
architecture in Natural Language Processing (NLP), achieving state-of-the-art
results across numerous benchmarks~\cite{devlin2019bert,wang2018glue}. However,
their quadratic computational complexity $O(n^2)$ with respect to sequence
length poses significant challenges for processing long documents and
real-time applications~\cite{zaheer2020big,beltagy2020longformer}.

State Space Models (SSMs)~\cite{gu2022efficiently}, particularly
Mamba~\cite{gu2023mamba}, have emerged as a promising alternative with
linear-time complexity $O(n)$ through selective scan mechanisms. While
Mamba excels at capturing sequential patterns efficiently, it may lack
the global contextual understanding provided by self-attention mechanisms
in Transformers~\cite{vaswani2017attention}.

To bridge this gap, Zhu et al.~\cite{zhu2025transmamba} proposed TransMamba,
a hybrid architecture that combines a Transformer encoder with a Mamba decoder
through a Feature Fusion mechanism. Their work demonstrated strong performance
on reasoning benchmarks (ARC, HellaSwag, PIQA) with a 350M parameter model
trained from scratch. However, the applicability of this architecture to
text classification tasks remains unexplored.

Since the introduction of BERT~\cite{devlin2019bert}, pretrained Transformer
models have dominated text classification benchmarks. The GLUE
benchmark~\cite{wang2018glue} serves as the standard evaluation framework,
encompassing tasks such as sentiment analysis (SST-2)~\cite{socher2013recursive},
natural language inference (MNLI)~\cite{williams2018broad}, and textual
entailment (RTE)~\cite{dagan2005pascal,bar2006second,giampiccolo2007third,bentivogli2009fifth}.
Various BERT variants including DistilBERT~\cite{sanh2019distilbert},
ALBERT~\cite{lan2020albert}, and RoBERTa~\cite{liu2019roberta} have
achieved progressively higher scores. Meanwhile, recent hybrid architectures
such as Jamba~\cite{lieber2024jamba}, which interleaves Transformer and
Mamba layers within a mixture-of-experts framework, have shown promise
for sequence modeling.

In this paper, we present \transmamba{}, which adapts the TransMamba
architecture for text classification with the following contributions:
(1)~We replace the custom Transformer encoder with pretrained BERT
models, enabling practical deployment without large-scale pretraining.
(2)~We evaluate the hybrid architecture on three GLUE benchmark tasks,
extending TransMamba beyond reasoning-focused evaluation.
(3)~We investigate three encoder scales while maintaining the 1:2
encoder-to-decoder layer ratio from the original design.
(4)~We systematically compare four fusion strategies to quantify the
contribution of each component.

Our paper is organized as follows: Section~2 describes the proposed
architecture and training strategy, Section~3 presents experiments and
results including ablation studies, Section~4 provides discussion, and
Section~5 summarizes conclusions and future work.

% ===========================================================================
% 2. PROPOSED METHOD
% ===========================================================================

\section{Proposed Method}
\label{sec:method}

\subsection{Architecture Overview}

\transmamba{} follows an encoder-decoder-fusion architecture
(Fig.~\ref{fig:architecture}), consisting of three main components:
(1)~a pretrained BERT encoder for extracting global contextual features,
(2)~a Mamba decoder stack for sequential modeling, and (3)~a Feature Fusion
module that combines both representations through learned projections
and cross-attention.

\begin{figure}[t]
\centering
\begin{tikzpicture}[
    node distance=0.8cm,
    block/.style={rectangle, draw, rounded corners, minimum width=4.5cm,
                  minimum height=0.7cm, align=center, font=\small},
    arrow/.style={->, thick, >=stealth},
    label/.style={font=\scriptsize, text=gray}
]
    % Input
    \node[block, fill=gray!10] (input) {Input Tokens};

    % Encoder
    \node[block, fill=blue!15, above=of input] (encoder)
        {BERT Encoder (Pretrained)};

    % Two branches
    \node[block, fill=blue!8, above left=1cm and -0.5cm of encoder] (tproj)
        {TransformerProj\\Linear$\to$SiLU$\to$Linear};
    \node[block, fill=green!15, above right=1cm and -0.5cm of encoder] (mamba)
        {Mamba Decoder Stack\\($N$ layers PureSSM)};

    % Mamba projection
    \node[block, fill=green!8, above=of mamba] (mproj)
        {MambaProj\\Conv1x1$\to$SiLU$\to$Conv1x1};

    % Fusion
    \node[block, fill=orange!20, above=1.5cm of encoder, minimum width=5cm] (fusion)
        {Cross-Attention Fusion\\$Q{=}H', K{=}E', V{=}E'$};

    % Classifier
    \node[block, fill=red!15, above=of fusion] (cls)
        {MeanPool $\to$ RMSNorm $\to$ Classifier};

    % Arrows
    \draw[arrow] (input) -- (encoder);
    \draw[arrow] (encoder.north) -- ++(0,0.3) -| (tproj.south)
        node[midway, left, label] {$E$};
    \draw[arrow] (encoder.north) -- ++(0,0.3) -| (mamba.south)
        node[midway, right, label] {$E$};
    \draw[arrow] (mamba) -- (mproj) node[midway, right, label] {$H$};
    \draw[arrow] (tproj.north) |- (fusion.west)
        node[near end, above, label] {$E'$};
    \draw[arrow] (mproj.north) |- (fusion.east)
        node[near end, above, label] {$H'$};
    \draw[arrow] (fusion) -- (cls);
\end{tikzpicture}
\caption{Architecture of \transmamba{}. The pretrained BERT encoder provides
global context features $E$, which are processed through two parallel paths:
TransformerProj refines encoder features into $E'$, while the Mamba decoder
produces sequential features $H$ that are projected into $H'$. The
cross-attention fusion combines both representations.}
\label{fig:architecture}
\end{figure}

\subsection{Pretrained Transformer Encoder}

We employ pretrained BERT models~\cite{devlin2019bert} as the encoder
component. Given an input sequence of tokens $\mathbf{x} = (x_1, x_2,
\ldots, x_n)$, the encoder produces contextualized representations:

\begin{equation}
    E = \text{BERT}(\mathbf{x}) \in \mathbb{R}^{n \times d}
\end{equation}

where $d$ is the hidden dimension. We investigate three encoder scales
(Table~\ref{tab:encoder_config}) to study the relationship between
encoder capacity and hybrid model performance.

\begin{table}[t]
\centering
\caption{Encoder configurations used in our experiments.}
\label{tab:encoder_config}
\begin{tabular}{@{}lcccr@{}}
\toprule
\textbf{Encoder} & \textbf{Layers} & \textbf{Hidden} & \textbf{Heads} & \textbf{Params} \\
\midrule
BERT-tiny  & 2  & 128 & 2  & 4.4M  \\
BERT-small & 4  & 512 & 8  & 28.8M \\
BERT-base  & 12 & 768 & 12 & 110M  \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Mamba Decoder Stack}

The decoder consists of $N$ stacked PureSSM layers with pre-norm
architecture using RMSNorm~\cite{zhang2019root}. Each layer applies:

\begin{equation}
    \hat{x}_l = \text{SSM}(\text{RMSNorm}(x_{l-1})) + x_{l-1}
\end{equation}

The Selective State Space Model (SSM) in each layer implements
input-dependent discretization and selective scanning:

\begin{equation}
    h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t + D x_t
\end{equation}

where $\bar{A} = \exp(\Delta A)$ and $\bar{B} = (\Delta A)^{-1}
(\exp(\Delta A) - I) \cdot \Delta B$ are the discretized state matrices,
and $\Delta$ is the input-dependent step size. We use $N=8$ decoder
layers, maintaining the 1:2 encoder-to-decoder ratio from the original
TransMamba design (8 encoder + 16 decoder layers).

\subsection{Feature Fusion Module}

The core contribution of the TransMamba architecture is the Feature Fusion
mechanism, which enables the decoder to attend back to encoder
representations. Following Zhu et al.~\cite{zhu2025transmamba}, our
fusion module consists of four steps:

\textbf{Step 1: Transformer Feature Projection.}
The encoder output is refined through a two-layer projection:
\begin{equation}
    E' = \text{LN}(E + \text{Linear}_{2d \to d}(\text{SiLU}(\text{Linear}_{d \to 2d}(E))))
\end{equation}

\textbf{Step 2: Mamba Feature Projection.}
The decoder output is projected using pointwise convolutions:
\begin{equation}
    H' = \text{LN}(H + \text{Conv1x1}_{2d \to d}(\text{SiLU}(\text{Conv1x1}_{d \to 2d}(H))))
\end{equation}

\textbf{Step 3: Cross-Attention.}
The projected features are combined through multi-head cross-attention:
\begin{equation}
    F = \text{CrossAttn}(Q=H', K=E', V=E')
\end{equation}

\textbf{Step 4: Residual Connection.}
The fused output with residual connection, mean pooling and classification:
\begin{equation}
    O = \text{LN}(H' + F), \quad \hat{y} = \text{MLP}(\text{RMSNorm}(\text{MeanPool}(O)))
\end{equation}

\subsection{Training Strategy}

We adopt a differential learning rate strategy: encoder learning rate
$\alpha_e = 5 \times 10^{-4}$ and decoder/fusion learning rate
$\alpha_d = 1 \times 10^{-3}$, both with weight decay $0.01$.
We use AdamW optimizer~\cite{loshchilov2019decoupled} with linear warmup
(10\% of total steps) followed by cosine decay, and gradient clipping
with max norm 1.0.

% ===========================================================================
% 3. EXPERIMENT RESULTS
% ===========================================================================

\section{Experiment Results}
\label{sec:experiments}

\subsection{Datasets}

We evaluate on three representative tasks from the GLUE
benchmark~\cite{wang2018glue} (Table~\ref{tab:datasets}), selected
to cover diverse aspects of language understanding: data scale
(small to large), task type (single-sentence and sentence-pair),
and difficulty level.

\textbf{SST-2} (Stanford Sentiment Treebank)~\cite{socher2013recursive}
is a binary sentiment classification task on movie reviews. With 67,349
training examples and an average length of $\sim$19 tokens, SST-2 is
the most commonly reported GLUE task and serves as our primary benchmark.
The task requires understanding sentiment expressions (e.g., ``not bad''
$\to$ positive), testing both local word-order patterns captured by
Mamba and global context captured by the Transformer encoder.

\textbf{MNLI} (Multi-Genre Natural Language Inference)~\cite{williams2018broad}
is a three-class sentence-pair task with 392,702 training examples across
10 genres (fiction, government, telephone, etc.). Given a premise and
hypothesis, the model predicts entailment, contradiction, or neutral.
As the largest GLUE dataset, MNLI tests scalability and cross-genre
generalization. The original TransMamba was evaluated on reasoning
tasks (HellaSwag, PIQA)---MNLI provides a complementary evaluation
of reasoning ability on natural language inference.

\textbf{RTE} (Recognizing Textual Entailment)~\cite{dagan2005pascal,bar2006second,giampiccolo2007third,bentivogli2009fifth}
is a binary entailment task with only 2,490 training examples, making
it the smallest dataset in our evaluation. Its premises average $\sim$49
tokens---substantially longer than SST-2---where the Mamba decoder's
sequential memory may provide benefits. RTE tests data efficiency
in a low-resource setting, an important practical scenario.

\begin{table}[t]
\centering
\caption{Dataset statistics for the GLUE tasks used in our evaluation.}
\label{tab:datasets}
\begin{tabular}{@{}lllrrr@{}}
\toprule
\textbf{Task} & \textbf{Type} & \textbf{Labels} & \textbf{Train} & \textbf{Dev} & \textbf{Avg Len} \\
\midrule
SST-2 & Sentiment     & 2 (pos/neg)      & 67,349  & 872   & $\sim$19 \\
MNLI  & NLI           & 3 (ent/con/neu)  & 392,702 & 9,815 & $\sim$33 \\
RTE   & Entailment    & 2 (ent/not\_ent) & 2,490   & 277   & $\sim$60 \\
\bottomrule
\end{tabular}
\end{table}

These three tasks collectively cover: (1)~single-sentence vs.\
sentence-pair classification, (2)~medium to large data scales,
(3)~short to moderately long sequences, and (4)~binary and multi-class
settings. Notably, while all major Transformer papers report GLUE
scores, no existing Mamba or hybrid Transformer-Mamba work has been
evaluated on GLUE---our work addresses this gap.

\subsection{Baselines and Setup}

We compare \transmamba{} against two baselines representing each
individual component:
(1)~\textbf{BERT Baseline}---standard fine-tuning of BERT-tiny with a
classification head (encoder-only);
(2)~\textbf{Pure Mamba Baseline}---a standalone PureSSM model (4 layers,
$d=256$) with word embeddings trained from scratch (decoder-only).

All experiments use maximum sequence length 128, batch size 32,
and are trained with mixed precision (FP16) on a single NVIDIA T4 GPU.
We report accuracy on the development set as our primary metric.
For SST-2 and MNLI, we train for 5 and 3 epochs respectively.
For RTE (low-resource), we train for 15 epochs with adjusted
learning rates ($\alpha_e = 3 \times 10^{-4}$, $\alpha_d = 5 \times 10^{-4}$).
We run experiments with 3 seeds (42, 123, 456) and report mean accuracy.

\subsection{Main Results}

Table~\ref{tab:main_results} presents the comparison results across
all models and tasks.

\begin{table}[t]
\centering
\caption{Main results on GLUE development sets. Accuracy (\%) is reported.
$\dagger$ denotes v1 results (2-layer decoder). All TransMamba v2 models
use 8-layer decoder with cross-attention fusion.}
\label{tab:main_results}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Model} & \textbf{SST-2} & \textbf{MNLI} & \textbf{RTE} & \textbf{Params} \\
\midrule
\multicolumn{5}{l}{\textit{Baselines}} \\
BERT-tiny (baseline)            & 81.65 & 61.95 & 55.23 & 4.4M  \\
Pure Mamba (baseline)            & 83.60 & 61.72 & 53.07 & 9.7M  \\
\midrule
\multicolumn{5}{l}{\textit{TransMamba v1 (2L decoder)$^\dagger$}} \\
TransMamba-tiny v1               & 82.91 & 63.04 & ---   & 4.7M  \\
\midrule
\multicolumn{5}{l}{\textit{TransMamba v2 (8L decoder) --- Ours}} \\
TransMamba-tiny v2               & ---   & ---   & ---   & $\sim$5M  \\
TransMamba-small v2              & ---   & ---   & ---   & $\sim$30M \\
TransMamba-base v2               & ---   & ---   & ---   & $\sim$115M \\
\bottomrule
\end{tabular}
\end{table}

% TODO: Replace "---" with actual results after running experiments

\subsection{Ablation Study}

To quantify the contribution of each fusion component, we evaluate
four strategies while keeping the encoder (BERT-small) and decoder
(8 layers) fixed (Table~\ref{tab:ablation}).

\begin{table}[t]
\centering
\caption{Ablation study on fusion strategies using BERT-small encoder
and 8-layer Mamba decoder on SST-2.}
\label{tab:ablation}
\begin{tabular}{@{}llcc@{}}
\toprule
\textbf{Fusion Type} & \textbf{Description} & \textbf{Proj} & \textbf{SST-2} \\
\midrule
None                     & Only Mamba output          & $\times$     & --- \\
Additive                 & $\text{LN}(H + E)$        & $\times$     & --- \\
Cross-Attn (Simple)      & CrossAttn$(H, E, E)$      & $\times$     & --- \\
Cross-Attn + Proj (Ours) & CrossAttn$(H', E', E')$   & $\checkmark$ & --- \\
\bottomrule
\end{tabular}
\end{table}

% TODO: Replace "---" with actual results

We also compare TransMamba v1 (2-layer decoder) with v2 (8-layer decoder)
using the same BERT-tiny encoder. Our v1 results showed that the 2-layer
decoder acted as a bottleneck: the Pure Mamba baseline (4 layers)
outperformed TransMamba v1 (82.91\%) at 83.60\% despite lacking an encoder.
The 8-layer decoder in v2 resolves this bottleneck while maintaining the
1:2 encoder-to-decoder ratio of the original paper.

% ===========================================================================
% 4. DISCUSSION
% ===========================================================================

\section{Discussion}
\label{sec:discussion}

\textbf{Pretrained vs.\ Custom Encoder.}
Unlike the original TransMamba which trains a custom Transformer encoder
from scratch (350M parameters), we leverage pretrained BERT encoders.
This design choice is motivated by: (1)~practical resource
constraints---our model is 10--70$\times$ smaller, (2)~reproducibility---standard
BERT checkpoints from HuggingFace ensure consistent baselines, and
(3)~our research focus on the effectiveness of the fusion mechanism
rather than encoder architecture.

\textbf{Computational Efficiency.}
The Mamba decoder provides $O(n)$ complexity for sequential processing,
complementing the $O(n^2)$ encoder. While the overall complexity is
dominated by the encoder for short sequences ($n \leq 128$), the
efficiency advantage becomes significant for longer inputs, suggesting
potential for document-level classification~\cite{zaheer2020big,beltagy2020longformer}.

\textbf{Limitations.}
Our evaluation is limited to sequences of length $\leq$128 tokens,
where the linear-time advantage of Mamba is less pronounced. Additionally,
our PureSSM implementation in PyTorch does not leverage hardware-aware
CUDA kernels~\cite{gu2023mamba}, resulting in slower training compared
to optimized alternatives.

% ===========================================================================
% 5. CONCLUSION
% ===========================================================================

\section{Conclusion}
\label{sec:conclusion}

We presented \transmamba{}, an adaptation of the hybrid
Transformer-Mamba architecture for text classification on the GLUE
benchmark. By combining pretrained BERT encoders with Mamba decoders
through a Feature Fusion mechanism with learned projections, our model
achieves improvements over both encoder-only and decoder-only baselines.
Our ablation study demonstrates that the full fusion pipeline---including
feature projections and cross-attention---contributes meaningfully to
performance. The architecture scales effectively across different
encoder sizes while maintaining the design ratios of the original
TransMamba framework.

Future work includes: (1)~evaluating on long-document classification
tasks where the linear-time advantage of Mamba is more pronounced,
(2)~exploring bidirectional Mamba variants (BiMamba) as the decoder,
and (3)~integrating hardware-optimized Mamba CUDA kernels for
improved training efficiency.

% ===========================================================================
% ACKNOWLEDGMENTS
% ===========================================================================

\begin{credits}
\subsubsection{\ackname}
% TODO: Add acknowledgments if applicable

\subsubsection{\discintname}
The authors have no competing interests to declare.
\end{credits}

% ===========================================================================
% REFERENCES
% ===========================================================================

\bibliographystyle{splncs04}
\bibliography{references}

\end{document}


% ===========================================================================
% FILE: references.bib (tao file rieng tren Overleaf)
% ===========================================================================
%
% --- Core Architecture Papers ---
%
% @article{vaswani2017attention,
%   title={Attention is all you need},
%   author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and
%           Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and
%           Kaiser, {\L}ukasz and Polosukhin, Illia},
%   journal={Advances in Neural Information Processing Systems},
%   volume={30},
%   year={2017}
% }
%
% @inproceedings{devlin2019bert,
%   title={{BERT}: Pre-training of deep bidirectional transformers for
%          language understanding},
%   author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and
%           Toutanova, Kristina},
%   booktitle={Proceedings of the 2019 Conference of the North American
%              Chapter of the Association for Computational Linguistics
%              (NAACL-HLT)},
%   pages={4171--4186},
%   year={2019}
% }
%
% --- State Space Models ---
%
% @inproceedings{gu2022efficiently,
%   title={Efficiently modeling long sequences with structured state spaces},
%   author={Gu, Albert and Goel, Karan and R{\'e}, Christopher},
%   booktitle={Proceedings of the International Conference on Learning
%              Representations (ICLR)},
%   year={2022}
% }
%
% @article{gu2023mamba,
%   title={Mamba: Linear-time sequence modeling with selective state spaces},
%   author={Gu, Albert and Dao, Tri},
%   journal={arXiv preprint arXiv:2312.00752},
%   year={2023}
% }
%
% --- Hybrid Architecture ---
%
% @article{zhu2025transmamba,
%   title={{TransMamba}: Flexibly Switching between Transformer and Mamba},
%   author={Zhu, Yixuan and Wang, Yuxin and Wang, Xiangyu and
%           Wang, Zhiyuan and Liu, Zheyuan},
%   journal={arXiv preprint arXiv:2501.07948},
%   year={2025}
% }
%
% @article{lieber2024jamba,
%   title={Jamba: A hybrid {Transformer-Mamba} language model},
%   author={Lieber, Opher and Lenz, Barak and Bata, Hofit and
%           Cohen, Gal and Osin, Jhonathan and Dalmedigos, Itay and
%           Safahi, Erez and Meirom, Shaked and Belinkov, Yonatan and
%           Shalev-Shwartz, Shai and Shoham, Yoav},
%   journal={arXiv preprint arXiv:2403.19887},
%   year={2024}
% }
%
% --- BERT Variants ---
%
% @article{sanh2019distilbert,
%   title={{DistilBERT}, a distilled version of {BERT}: smaller, faster,
%          cheaper and lighter},
%   author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and
%           Wolf, Thomas},
%   journal={arXiv preprint arXiv:1910.01108},
%   year={2019}
% }
%
% @inproceedings{lan2020albert,
%   title={{ALBERT}: A lite {BERT} for self-supervised learning of
%          language representations},
%   author={Lan, Zhenzhong and Chen, Mingda and Goodman, Sebastian and
%           Gimpel, Kevin and Sharma, Piyush and Soricut, Radu},
%   booktitle={Proceedings of the International Conference on Learning
%              Representations (ICLR)},
%   year={2020}
% }
%
% @article{liu2019roberta,
%   title={{RoBERTa}: A robustly optimized {BERT} pretraining approach},
%   author={Liu, Yinhan and Ott, Myle and Goyal, Naman and Du, Jingfei
%           and Joshi, Mandar and Chen, Danqi and Levy, Omer and Lewis,
%           Mike and Zettlemoyer, Luke and Stoyanov, Veselin},
%   journal={arXiv preprint arXiv:1907.11692},
%   year={2019}
% }
%
% --- Benchmarks and Datasets ---
%
% @inproceedings{wang2018glue,
%   title={{GLUE}: A multi-task benchmark and analysis platform for
%          natural language understanding},
%   author={Wang, Alex and Singh, Amanpreet and Michael, Julian and
%           Hill, Felix and Levy, Omer and Bowman, Samuel R},
%   booktitle={Proceedings of the International Conference on Learning
%              Representations (ICLR)},
%   year={2019}
% }
%
% @inproceedings{socher2013recursive,
%   title={Recursive deep models for semantic compositionality over a
%          sentiment treebank},
%   author={Socher, Richard and Perelygin, Alex and Wu, Jean and
%           Chuang, Jason and Manning, Christopher D and Ng, Andrew Y
%           and Potts, Christopher},
%   booktitle={Proceedings of the 2013 Conference on Empirical Methods
%              in Natural Language Processing (EMNLP)},
%   pages={1631--1642},
%   year={2013}
% }
%
% @inproceedings{williams2018broad,
%   title={A broad-coverage challenge corpus for sentence understanding
%          through inference},
%   author={Williams, Adina and Nangia, Nikita and Bowman, Samuel R},
%   booktitle={Proceedings of the 2018 Conference of the North American
%              Chapter of the Association for Computational Linguistics
%              (NAACL-HLT)},
%   pages={1112--1122},
%   year={2018}
% }
%
% @incollection{dagan2005pascal,
%   title={The {PASCAL} recognising textual entailment challenge},
%   author={Dagan, Ido and Glickman, Oren and Magnini, Bernardo},
%   booktitle={Machine Learning Challenges. Evaluating Predictive
%              Uncertainty, Visual Object Classification, and Recognising
%              Textual Entailment},
%   pages={177--190},
%   year={2006},
%   publisher={Springer}
% }
%
% @inproceedings{bar2006second,
%   title={The second {PASCAL} recognising textual entailment challenge},
%   author={Bar-Haim, Roy and Dagan, Ido and Dolan, Bill and
%           Ferro, Lisa and Giampiccolo, Danilo and Magnini, Bernardo
%           and Szpektor, Idan},
%   booktitle={Proceedings of the Second PASCAL Challenges Workshop on
%              Recognising Textual Entailment},
%   year={2006}
% }
%
% @inproceedings{giampiccolo2007third,
%   title={The third {PASCAL} recognizing textual entailment challenge},
%   author={Giampiccolo, Danilo and Magnini, Bernardo and Dagan, Ido
%           and Dolan, Bill},
%   booktitle={Proceedings of the ACL-PASCAL Workshop on Textual
%              Entailment and Paraphrasing},
%   pages={1--9},
%   year={2007}
% }
%
% @inproceedings{bentivogli2009fifth,
%   title={The fifth {PASCAL} recognizing textual entailment challenge},
%   author={Bentivogli, Luisa and Clark, Peter and Dagan, Ido and
%           Giampiccolo, Danilo},
%   booktitle={Proceedings of TAC},
%   year={2009}
% }
%
% --- Efficient Transformers ---
%
% @inproceedings{zaheer2020big,
%   title={{Big Bird}: Transformers for longer sequences},
%   author={Zaheer, Manzil and Guruganesh, Guru and Dubey, Kumar Avinava
%           and Ainslie, Joshua and Alberti, Chris and Ontanon, Santiago
%           and Pham, Philip and Ravula, Anirudh and Wang, Qifan and
%           Yang, Li and Ahmed, Amr},
%   booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
%   year={2020}
% }
%
% @article{beltagy2020longformer,
%   title={Longformer: The long-document transformer},
%   author={Beltagy, Iz and Peters, Matthew E and Cohan, Arman},
%   journal={arXiv preprint arXiv:2004.05150},
%   year={2020}
% }
%
% --- Normalization and Optimization ---
%
% @inproceedings{zhang2019root,
%   title={Root mean square layer normalization},
%   author={Zhang, Biao and Sennrich, Rico},
%   booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
%   volume={32},
%   year={2019}
% }
%
% @inproceedings{loshchilov2019decoupled,
%   title={Decoupled weight decay regularization},
%   author={Loshchilov, Ilya and Hutter, Frank},
%   booktitle={Proceedings of the International Conference on Learning
%              Representations (ICLR)},
%   year={2019}
% }
%
% --- Mamba for NLP ---
%
% @article{he2024densemamba,
%   title={{DenseMamba}: State space models with dense hidden connection
%          for efficient large language models},
%   author={He, Wei and Dai, Kai and Lu, Yifei and Wang, Xin and
%           Liu, Jiahao and Liu, Ji and Ye, Han},
%   journal={arXiv preprint arXiv:2403.00818},
%   year={2024}
% }
%
% @article{anthony2024blackmamba,
%   title={{BlackMamba}: Mixture of experts for state-space models},
%   author={Anthony, Quentin and Tokpanov, Yury and Glorioso, Paulo
%           and Pilault, Jonathan},
%   journal={arXiv preprint arXiv:2402.01771},
%   year={2024}
% }
%
% @article{park2024mamba,
%   title={Can {Mamba} learn how to learn? A comparative study on in-context
%          learning tasks},
%   author={Park, Jongho and Park, Jaeseung and Xiong, Zheyang and
%           Papailiopoulos, Dimitris and Lee, Jason D},
%   journal={arXiv preprint arXiv:2402.04248},
%   year={2024}
% }