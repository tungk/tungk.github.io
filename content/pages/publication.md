+++
title = 'Publication'
draft = false
+++

## Selected Publication

* Hong-Phuc Phan, Tuan-Anh Vu, **Tung Kieu**, Son Ha Xuan, Sean Bin Yang, Bin Yang, and Christian S. Jensen. *AMP-OD: Active Model Probing for Unsupervised Outlier Model Selection*. Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (**KDD**), 2027, \[**CORE A**\*]. {{< details summary="See the details" >}}Outlier detection is often deployed in fully unsupervised settings, where labels are unavailable and practitioners must choose a detector without direct performance feedback. Existing unsupervised outlier model selection methods predict detector quality from static meta-features and historical performance on meta-datasets, but they commit before observing how any detector behaves on the target dataset. We propose AMP-OD, an execution-conditioned probe-then-commit framework. AMP-OD runs a small number of inexpensive probe detectors, uses their score patterns as target-dataset evidence, and then selects a final detector from a larger candidate pool. To interpret unlabeled probe responses, we introduce a Leave-One-Out Family Probe Posterior (LOO-FPP), which converts probe score patterns into gated detector-family evidence while reducing self-validation in pseudo-label construction. AMP-OD further combines this evidence with historical reliability through a warm-start prior, a sequential probe/stop policy, and lower-confidence-bound arbitration. Extensive experiments and ablation studies show that detector responses provide useful evidence beyond static meta-features, with the LOO-FPP and conservative arbitration as key contributors.{{< /details >}}
* Xuan-Thong Truong, Trung-Kien Le, **Tung Kieu**, Thi-Thu Nguyen, and Nhat-Hai Nguyen.
*ALER-TI: Aligned Latent Embedding Retrieval for Time Series Imputation*. Proceedings of the ACM International Conference on Web Search and Data Mining (**WSDM**), 2027, \[**CORE A**]. {{< details summary="See the details" >}}Deep learning has significantly advanced time series imputation, yet most existing architectures primarily rely on localized temporal context within the corrupted input sequence. This reliance can be limiting in real-world scenarios, where time series often exhibit non-stationary dynamics, weak temporal correlations, and infrequent patterns that are difficult to reconstruct from nearby observations alone. In this paper, we propose ALER-TI, Aligned Latent Embedding Retrieval for Time Series Imputation, a retrieval-augmented framework that explicitly leverages historical patterns to supplement degraded local context for more reliable missing-value reconstruction. The core of ALER-TI is Latent Embedding Alignment (LEA), which mitigates the representation mismatch between corrupted queries and complete historical candidates. By applying post-hoc masking in the latent space, LEA aligns candidates with the query's missingness pattern while allowing historical embeddings to be pre-computed and cached for efficient retrieval. ALER-TI is model-agnostic and can be integrated with various imputation backbones through a lightweight adaptation module. Extensive experiments on six real-world datasets under different missing rates demonstrate that ALER-TI consistently improves strong baseline models and enhances robustness across diverse imputation settings.{{< /details >}}
* Tuan-Binh Tran, Thanh Tam Nguyen, Quoc Viet Hung Nguyen, Dung D. Le, **Tung Kieu**, and Thanh Trung Huynh.
*EPIC: Explicit Posterior Item Conditioning for Semantic ID Diffusion Recommendation*. Proceedings of the ACM International Conference on Web Search and Data Mining (**WSDM**), 2027, \[**CORE A**]. {{< details summary="See the details" >}}Semantic ID (SID) generative recommendation predicts the next item by generating a short tuple of discrete tokens. Recent masked-diffusion methods improve this process through bidirectional context and flexible decoding, yet recommendation ultimately requires selecting among complete catalog items. At each denoising step, a partial SID can correspond to multiple feasible items, while existing methods primarily reason through position-wise token predictions. We propose Explicit Posterior Item Conditioning (EPIC), which introduces explicit item-level competition into SID denoising. EPIC constructs a personalized posterior over feasible candidate items using the current generation context and the user's recent interactions, then projects this distribution back to unresolved SID positions to guide subsequent token decisions. The pretrained backbone remains frozen and requires no additional decoder forward pass. Experiments on four Amazon benchmarks show consistent improvements over strong baselines, while diagnostic analyses indicate that the gains primarily arise from personalized transition evidence that preserves promising item hypotheses during denoising.{{< /details >}}
* Kien Le Trung, **Tung Kieu**, Dinh Duc Nha Nguyen, Cuong Do, Bin Yang, and Phi Long Nguyen.
*AeroChem: Closed-loop Physics-Informed State Space Modeling for Long-term Chemically-Reactive Air Quality Forecasting*.
Proceedings of the Advances in Neural Information Processing Systems (**NEURIPS**), 2026, \[**CORE A**\*]. {{< details summary="See the details" >}}Accurate long-term air quality forecasting is challenging due to long-range transport, meteorological variability, and nonlinear chemical reactions underlying secondary pollutant formation. Purely data-driven models often suffer from long-horizon drift and lack physical consistency, while existing physics-informed approaches commonly follow open-loop designs that fuse neural and physical components only at the output level. We propose AeroChem, a closed-loop physics-informed state space framework for chemically reactive air quality forecasting. AeroChem adopts a two-branch co-correction design: a UDE-based process branch provides a structured but imperfect physical prior, while a Mamba-based virtual measurement branch captures long-range temporal context and produces probabilistic virtual observations. Neural Kalman Fusion couples the two branches by correcting the latent physical state at each forecasting step and feeding the posterior state back into the simulator, reducing long-horizon drift. We further introduce HanoiAir, a real-world dataset integrating pollutant concentrations, meteorology, and emission inventories. Experiments on HanoiAir and a public benchmark show that AeroChem improves long-horizon stability and sudden-change forecasting.{{< /details >}}
* Jicheng Yuan, **Tung Kieu**, Duc Manh Nguyen, Manfred Hauswirth, and Danh Le-Phuoc.
*Contrastive Hypergraph Source-free Domain Adaptive Object Detection in Adverse Weathers*.
Proceedings of the Advances in Neural Information Processing Systems (**NEURIPS**), 2026, \[**CORE A**\*]. {{< details summary="See the details" >}}Adverse weather conditions such as fog, rain, snow, and low illumination introduce structured visibility degradations that severely impair object detectors trained on clear-weather data. When source data are inaccessible due to privacy or trans- mission constraints, source-free domain adaptation (SFDA) becomes necessary. However, existing SFDA methods based on pairwise contrastive learning struggle to capture high-order semantics and often suffer from agreement collapse under severe corruption. We propose REUNION, a hypergraph-guided SFDA framework for object detection in adverse weather conditions. REUNION constructs a contrastive hypergraph over target-domain object proposals, encoding high-order relations through intra-image context, weather-aware grouping, prototype-based semantic anchors, and uncertainty-aware connections between reliable and low-confidence instances. To effectively exploit these structured relations, we introduce a group-wise Hyper-InfoNCE objective that optimizes representations at the hyperedge level, enabling semantic information to propagate from confident proposals to corrupted or low-contrast instances. Experiments with diverse benchmarks demonstrate that REUNION effectively mitigates domain shifts and achieves state-of-the-art performance in SFDA settings.{{< /details >}}
* Sean Bin Yang, Ying Sun, Zongyi Xu, **Tung Kieu**, Jilin Hu, Bin Yang, Kristian Torp, Hua Lu, and Torben Bach Pedersen.
*When Correlations Mislead: Confounder-Aware Multi-View Urban Region Representation Learning*.
Proceedings of the IEEE International Conference on Data Engineering (**ICDE**), 2027, \[**CORE A**\*].
* Zhe Li, Tengxue Zhang, Yang Shu, **Tung Kieu**, Lunting Fan, Yingying Zhang, and Bin Yang.
*BEACON: Bridging Multi-Modal Semantics and Asynchronous Propagation for Cloud System Anomaly Detection*.
Proceedings of the IEEE International Conference on Data Engineering (**ICDE**), 2027, \[**CORE A**\*].
* Trung Duc Anh Dang, **Tung Kieu**, and Sarah Masud.
*Scenario-based Probing and Steering Cultural Values in Large Language Models*.
Proceedings of the International Conference on Empirical Methods in Natural Language Processing (**EMNLP**), 2026, \[**CORE A**\*].
* Tuan-Binh Tran, Dat Nguyen Cong, Duc-Trong Le, Thanh Trung Huynh, and **Tung Kieu**.
*SCENARIODIFF: A Scenario-level Guidance Framework for Multimodal Time Series Forecasting*.
Proceedings of the IEEE International Conference on Data Mining (**ICDM**), 2026, \[**CORE A**\*].
* David Campos, Bin Yang, **Tung Kieu**, Lei Chen, Chenjuan Guo, and Christian S. Jensen.
*TimeBlocks: Versatile and Continual Time-Series Blockbase*.
Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (**KDD**), 2026, \[**CORE A**\*].
* Phuc Hong Phan, Anh Vu Tuan, **Tung Kieu**, Son Ha Xuan, Bin Yang, and Christian S. Jensen.
*Automatic Unsupervised Ensemble Outlier Model Selection*.
Proceedings of the International Conference on Machine Learning (**ICML**), 2026, \[**CORE A**\*].
* Dat Nguyen Cong, **Tung Kieu**, and Hoang Thanh-Tung.
*FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation*.
Proceedings of the Annual Meeting of the Association for Computational Linguistics (**ACL**), 2026, \[**CORE A**\*].
* Duc Kieu, Kien Do, Tuan Hoang, Thao Minh Le, **Tung Kieu**, Dang Nguyen, and Thin Nguyen.
*Universal Multi-Domain Translation via Diffusion Routers*.
Proceedings of the International Conference on Learning Representations (**ICLR**), 2026, \[**CORE A**\*].
* Taisei Hanyu, Nhat Chung, Huy Le, Toan Nguyen, Yuki Ikebe, Anthony Gunderman, Duy Nguyen Ho Minh, Khoa Vo, **Tung Kieu**, Kashu Yamazaki, Chase Rainwater, Anh Nguyen, and Ngan Le.
*SlotVLA: Towards Modeling of Object-Relation Representations in Robotic Manipulation*.
Proceedings of the IEEE International Conference on Robotics and Automation (**ICRA**), 2026, \[**CORE A**\*].
* Huy Le, Nhat Chung, **Tung Kieu**, Jingkang Yang, and Ngan Le.
*UNO: Unifying One-stage Video Scene Graph Generation via Object-Centric Visual Representation Learning*.
Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (**WACV**), 2026, \[**CORE A**].
* Nhat Chung, Taisei Hanyu, Toan Nguyen, Huy Le, Frederick Bumgarner, Duy Minh Ho Nguyen, Khoa Vo, Kashu Yamazaki, Chase Rainwater, **Tung Kieu**, Anh Nguyen, and Ngan Le.
*Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective*.
Proceedings of the Annual AAAI Conference on Artificial Intelligence (**AAAI**), 2026, \[**CORE A**\*].
* Buang Zhang, **Tung Kieu**, Xiangfei Qiu, Chenjuan Guo, Jilin Hu, Aoying Zhou, Christian S. Jensen, and Bin Yang.
*An Encode-then-Decompose Approach to Unsupervised Time Series Anomaly Detection on Contaminated Training Data*.
Proceedings of the IEEE International Conference on Data Engineering (**ICDE**), 2026, \[**CORE A**\*].
* Nghiem Thanh Pham, **Tung Kieu**, Duc-Manh Nguyen, Son Ha Xuan, Nghia Duong-Trung, and Danh Le-Phuoc.
*SLM-Bench: A Comprehensive Benchmark of Small Language Models on Environmental Impact*.
Proceedings of the International Conference on Empirical Methods in Natural Language Processing (**EMNLP**), 2025, \[**CORE A**\*].
* Huy Le, Nhat Chung, **Tung Kieu**, Anh Nguyen, and Ngan Le.
*BiMa: Towards Biases Mitigation for Text-Video Retrieval via Scene Element Guidance*.
Proceedings of the ACM International Conference on Multimedia (**MM**), 2025, \[**CORE A**\*].
* Duc Kieu, **Tung Kieu**, Peng Han, Bin Yang, Christian S. Jensen, and Bac Le.
*TEAM: Topological Evolution-aware Framework for Traffic Forecasting*.
Proceedings of the International Conference on Very Large Data Bases (**VLDB**), 2025, \[**CORE A**\*].
* David Campos, Bin Yang, **Tung Kieu**, Miao Zhang, Chenjuan Guo, and Christian S. Jensen.
*QCore: Data-Efficient, On-Device Continual Calibration for Quantized Models*.
Proceedings of the International Conference on Very Large Data Bases (**VLDB**), 2024, \[**CORE A**\*].
* Chenjuan Guo, Ronghui Xu, Yuan Ye, Bin Yang, **Tung Kieu**, Yan Zhao, and Christian S. Jensen.
*Efficient Stochastic Routing in Path-Centric Uncertain Road Networks*.
Proceedings of the International Conference on Very Large Data Bases (**VLDB**), 2024, \[**CORE A**\*].
* Huy Le, **Tung Kieu**, Anh Nguyen, and Ngan Le.
*WAVER: Writing-style Agnostic Video Retrieval via Distilling Vision-Language Models Through Open-Vocabulary Knowledge*.
Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (**ICASSP**), 2024, \[**CORE A**].
* David Campos, Miao Zhang, Bin Yang, **Tung Kieu**, Chenjuan Guo, and Christian S. Jensen.
*LightTS: Lightweight Time Series Classification with Adaptive Ensemble Distillation*.
Proceedings of the ACM SIGMOD International Conference on Management of Data (**SIGMOD**), 2023, \[**CORE A**\*].
* Razvan-Gabriel Cirstea, **Tung Kieu**, Chenjuan Guo, Xuanyi Dong, Shirui Pan, and Bin Yang.
*Triformer: Triangular, Variable-Specific Attention for Long Sequence Multivariate Time Series Forecasting*.
Proceedings of the International Joint Conference on Artificial Intelligence (**IJCAI**), 2022, \[**CORE A**\*].
* Razvan-Gabriel Cirstea, **Tung Kieu**, Chenjuan Guo, Shirui Pan, and Bin Yang.
*Towards Spatio-Temporal Aware Traffic Time Series Forecasting*.
Proceedings of the IEEE International Conference on Data Engineering (**ICDE**), 2022, \[**CORE A**\*].
* **Tung Kieu**, Bin Yang, Chenjuan Guo, Christian S. Jensen, Yan Zhao, Feiteng Huang, and Kai Zheng.
*Robust and Explainable Autoencoders for Time Series Outlier Detection*.
Proceedings of the IEEE International Conference on Data Engineering (**ICDE**), 2022, \[**CORE A**\*].
* **Tung Kieu**, Bin Yang, Chenjuan Guo, Razvan-Gabriel Cirstea, Yan Zhao, Yale Song, and Christian S. Jensen.
*Anomaly Detection in Time Series with Robust Variational Quasi-Recurrent Autoencoders*.
Proceedings of the IEEE International Conference on Data Engineering (**ICDE**), 2022, \[**CORE A**\*].
* Yan Zhao, Xuanhao Chen, Liwei Deng, **Tung Kieu**, Chenjuan Guo, Bin Yang, Kai Zheng, and Christian S. Jensen.
*Outlier Detection for Streaming Task Assignment in Crowdsourcing*.
Proceedings of the ACM The Web Conference (**WWW**), 2022, \[**CORE A**\*].
* David Campos, **Tung Kieu**, Chenjuan Guo, Feiteng Huang, Kai Zheng, Bin Yang, and Christian S. Jensen.
*Unsupervised Time Series Outlier Detection with Diversity-Driven Convolutional Ensembles*.
Proceedings of the International Conference on Very Large Data Bases (**VLDB**), 2022, \[**CORE A**\*].
* Razvan-Gabriel Cirstea, **Tung Kieu**, Chenjuan Guo, Bin Yang, and Sinno Jialin Pan.
*EnhanceNet: Plugin Neural Networks for Enhancing Correlated Time Series Forecasting*.
Proceedings of the IEEE International Conference on Data Engineering (**ICDE**), 2021, \[**CORE A**\*].
* **Tung Kieu**, Bin Yang, Chenjuan Guo, and Christian S. Jensen.
*Outlier Detection for Time Series with Recurrent Autoencoder Ensembles*.
Proceedings of the International Joint Conference on Artificial Intelligence (**IJCAI**), 2019, \[**CORE A**\*].
* **Tung Kieu**, Bin Yang, Chenjuan Guo, and Christian S. Jensen.
*Distinguishing Trajectories from Different Drivers using Incompletely Labeled Trajectories*.
Proceedings of the ACM International Conference on Information and Knowledge Management (**CIKM**), 2018, \[**CORE A**].
* **Tung Kieu**, Bin Yang, and Christian S. Jensen.
*Outlier Detection for Multidimensional Time Series Using Deep Neural Networks*.
Proceedings of the IEEE International Conference on Mobile Data Management (**MDM**), 2018, \[**CORE B**].
* Toan Nguyen, Duc Kieu, Bao Duong, **Tung Kieu**, Kien Do, Thin Nguyen, and Bac Le.
*Class-incremental Learning with Causal Relational Replay*.
**Expert Syst. Appl.**, 250, 2024.
* **Tung Kieu**, Bay Vo, Tuong Le, Zhi-Hong Deng, and Bac Le.
*Mining Top-k Co-occurrence Items with Sequential Pattern*.
**Expert Syst. Appl.**, 85, 2017.



## Preprint

* Hong-Phuc Phan, Tuan-Anh Vu, **Tung Kieu**, Son Ha Xuan, Sean Bin Yang, Bin Yang, and Christian S. Jensen. *AMP-OD: Active Model Probing for Unsupervised Outlier Model Selection--Extended Version*.
* Duc Kieu, Kien Do, Minh Khoa Le, **Tung Kieu**, Dang Nguyen, and Thin Nguyen.
*Modaliff: A Unified Token Diffusion Model for Any-to-Any Generation--Extended Version*.
* Nam V. Nguyen, Hoang Van Truong, Duy Khuong Nguyen, Quang Pham, and **Tung Kieu**.
*Agentic4PDM: Host-Governed Evidence Admissibility for Industrial Fault Diagnosis--Extended Version*.
* Phat T Tran-Truong, **Tung Kieu**, Xuan-Bach Le, Son X Ha, Duc Kieu, Bin Yang, and Christian S Jensen.
*SemReuse: Certified Reuse of Cached Natural-Language Views for Semantic Operators--Extended Version*.
* Kien Le Trung, **Tung Kieu**, Dinh Duc Nha Nguyen, Cuong Do, Bin Yang, and Phi Long Nguyen.
*AeroChem: Closed-loop Physics-Informed State Space Modeling for Long-term Chemically-Reactive Air Quality Forecasting--Extended Version*.
* Jicheng Yuan, **Tung Kieu**, Duc Manh Nguyen, Manfred Hauswirth, and Danh Le-Phuoc.
*Contrastive Hypergraph Source-free Domain Adaptive Object Detection in Adverse Weathers--Extended Version*.
* Sean Bin Yang, Ying Sun, Zongyi Xu, **Tung Kieu**, Jilin Hu, Bin Yang, Kristian Torp, Hua Lu, and Torben Bach Pedersen.
*When Correlations Mislead: Confounder-Aware Multi-View Urban Region Representation Learning--Extended Version*.
* Zhe Li, Tengxue Zhang, Yang Shu, **Tung Kieu**, Lunting Fan, Yingying Zhang, and Bin Yang.
*BEACON: Bridging Multi-Modal Semantics and Asynchronous Propagation for Cloud System Anomaly Detection--Extended Version*.
* Tuan-Binh Tran, Thanh Tam Nguyen, Quoc Viet Hung Nguyen, Dung D. Le, **Tung Kieu**, and Thanh Trung Huynh.
*EPIC: Explicit Posterior Item Conditioning for Semantic ID Diffusion Recommendation--Extended Version*.
* Tuan-Binh Tran, Dat Nguyen Cong, Duc-Trong Le, Thanh Trung Huynh, and **Tung Kieu**.
*SCENARIODIFF: A Scenario-level Guidance Framework for Multimodal Time Series Forecasting--Extended Version*.
* Xuan-Thong Truong, Trung-Kien Le, **Tung Kieu**, Thi-Thu Nguyen, and Nhat-Hai Nguyen.
*ALER-TI: Aligned Latent Embedding Retrieval for Time Series Imputation--Extended Version*.
* Trung Duc Anh Dang, **Tung Kieu**, and Sarah Masud.
*Scenario-based Probing and Steering Cultural Values in Large Language Models--Extended Version*.
* David Campos, Bin Yang, **Tung Kieu**, Lei Chen, Chenjuan Guo, and Christian S. Jensen.
*TimeBlocks: Versatile and Continual Time-Series Blockbase--Extended Version*.
* Phuc Hong Phan, Anh Vu Tuan, **Tung Kieu**, Son Ha Xuan, Bin Yang, and Christian S. Jensen.
*Automatic Unsupervised Ensemble Outlier Model Selection--Extended Version*.
* Dat Nguyen Cong, **Tung Kieu**, and Hoang Thanh-Tung.
*FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation--Extended Version*.
* Taisei Hanyu, Nhat Chung, Huy Le, Toan Nguyen, Yuki Ikebe, Anthony Gunderman, Duy Nguyen Ho Minh, Khoa Vo, **Tung Kieu**, Kashu Yamazaki, Chase Rainwater, Anh Nguyen, and Ngan Le.
*SlotVLA: Towards Modeling of Object-Relation Representations in Robotic Manipulation--Extended Version*.
* Nhat Chung, Taisei Hanyu, Toan Nguyen, Huy Le, Frederick Bumgarner, Duy Minh Ho Nguyen, Khoa Vo, Kashu Yamazaki, Chase Rainwater, **Tung Kieu**, Anh Nguyen, and Ngan Le.
*Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective--Extended Version*.
* Buang Zhang, **Tung Kieu**, Xiangfei Qiu, Chenjuan Guo, Jilin Hu, Aoying Zhou, Christian S. Jensen, and Bin Yang.
*An Encode-then-Decompose Approach to Unsupervised Time Series Anomaly Detection on Contaminated Training Data--Extended Version*.
* Duc Kieu, Kien Do, Tuan Hoang, Thao Minh Le, **Tung Kieu**, Dang Nguyen, and Thin Nguyen.
*Universal Multi-Domain Translation via Diffusion Routers--Extended Version*.
* Huy Le, Nhat Chung, **Tung Kieu**, Jingkang Yang, and Ngan Le.
*UNO: Unifying One-stage Video Scene Graph Generation via Object-Centric Visual Representation Learning--Extended Version*.
* Nghiem Thanh Pham, **Tung Kieu**, Duc-Manh Nguyen, Son Ha Xuan, Nghia Duong-Trung, and Danh Le-Phuoc.
*SLM-Bench: A Comprehensive Benchmark of Small Language Models on Environmental Impact--Extended Version*.
* Huy Le, Nhat Chung, **Tung Kieu**, Anh Nguyen, and Ngan Le.
*BiMa: Towards Biases Mitigation for Text-Video Retrieval via Scene Element Guidance--Extended Version*.
* Duc Kieu, **Tung Kieu**, Peng Han, Bin Yang, Christian S. Jensen, and Bac Le.
*TEAM: Topological Evolution-aware Framework for Traffic Forecasting--Extended Version*.
* Chenjuan Guo, Ronghui Xu, Bin Yang, Ye Yuan, **Tung Kieu**, Yan Zhao, and Christian S. Jensen.
*Efficient Stochastic Routing in Path-Centric Uncertain Road Networks--Extended Version*.
* David Campos, Bin Yang, **Tung Kieu**, Miao Zhang, Chenjuan Guo, and Christian S. Jensen.
*QCore: Data-Efficient, On-Device Continual Calibration for Quantized Models--Extended Version*.
* Huy Le, **Tung Kieu**, Anh Nguyen, and Ngan Le.
*WAVER: Writing-style Agnostic Video Retrieval via Distilling Vision-Language Models Through Open-Vocabulary Knowledge--Extended Version*.
* David Campos, Miao Zhang, Bin Yang, **Tung Kieu**, Chenjuan Guo, and Christian S. Jensen.
*LightTS: Lightweight Time Series Classification with Adaptive Ensemble Distillation--Extended Version*.
* **Tung Kieu**, Bin Yang, Chenjuan Guo, Christian S. Jensen, Yan Zhao, Feiteng Huang, and Kai Zheng. *Robust and Explainable Autoencoders for Time Series Outlier Detection--Extended Version*.
* Razvan-Gabriel Cirstea, **Tung Kieu**, Chenjuan Guo, Xuanyi Dong, Shirui Pan, and Bin Yang.
*Triformer: Triangular, Variable-Specific Attentions for Long Sequence Multivariate Time Series Forecasting--Extended Version*.
* Razvan-Gabriel Cirstea, **Tung Kieu**, Chenjuan Guo, Shirui Pan, and Bin Yang.
*Towards Spatio-Temporal Aware Traffic Time Series Forecasting--Extended Version*.
* David Campos, **Tung Kieu**, Chenjuan Guo, Feiteng Huang, Kai Zheng, Bin Yang, and Christian S. Jensen.
*Unsupervised Time Series Outlier Detection with Diversity-Driven Convolutional Ensembles--Extended Version*.
* Yan Zhao, Liwei Deng, Xuanhao Chen, Chenjuan Guo, Bin Yang, **Tung Kieu**, Feiteng Huang, Torben Bach Pedersen, Kai Zheng, and Christian S. Jensen.
*A Comparative Study on Unsupervised Anomaly Detection for Time Series: Experiments and Analysis--Extended Version*.

