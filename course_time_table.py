course_schedule = {
    "weeks": [
        {
            "week": 1,
            "title": "Natural Language Processing (NLP) I",
            "concepts": [
                "Introduction to NLP",
                "NLP in a clinical setting",
                "Medical terminology challenges",
                "Concept extraction from clinical notes",
                "Note summarization",
                "Clinical trial matching"
            ],
            "required_readings": [
                "Rajkomar, A., Dean, J., & Kohane, I. (2019). Machine learning in medicine. New England Journal of Medicine, 380(14), 1347-1358.",
                "Chapman, W. W., Nadkarni, P. M., Hirschman, L., et al. (2011). Overcoming barriers to NLP for clinical text. Journal of the American Medical Informatics Association, 18(5), 540-543."
            ]
        },
        {
            "week": 2,
            "title": "NLP II: Embeddings & Transformers",
            "concepts": [
                "Embeddings and their role in NLP",
                "Transformers and BERT",
                "Hugging Face library for NLP applications",
                "Clinical BERT and RNNs",
                "De-identification methods",
                "GPT-3 for medical question answering"
            ],
            "required_readings": [
                "Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL-HLT.",
                "Lee, J., Yoon, W., Kim, S., et al. (2020). BioBERT: a pre-trained biomedical language representation model. Bioinformatics, 36(4), 1234-1240."
            ]
        },
        {
            "week": 3,
            "title": "Generative AI",
            "concepts": [
                "Variational Autoencoders (VAEs)",
                "Generative Adversarial Networks (GANs)",
                "Healthcare applications of generative AI",
                "Synthetic data generation",
                "Data privacy concerns"
            ],
            "required_readings": [
                "Goodfellow, I., Pouget-Abadie, J., Mirza, M., et al. (2014). Generative adversarial nets. Advances in Neural Information Processing Systems.",
                "Gulrajani, I., & Lopez-Paz, D. (2020). In search of lost domain generalization. arXiv preprint arXiv:2007.01434."
            ]
        },
        {
            "week": 4,
            "title": "Medical Image Analysis I",
            "concepts": [
                "Introduction to computer vision in medicine",
                "Convolutional Neural Networks (CNNs)",
                "Image segmentation",
                "Medical image diagnosis",
                "Best practices for model evaluation",
                "PyTorch basics"
            ],
            "required_readings": [
                "Litjens, G., Kooi, T., Bejnordi, B. E., et al. (2017). A survey on deep learning in medical image analysis. Medical Image Analysis, 42, 60-88.",
                "Esteva, A., Kuprel, B., Novoa, R. A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542(7639), 115-118."
            ]
        },
        {
            "week": 5,
            "title": "Medical Image Analysis II",
            "concepts": [
                "Building, training, and evaluating models in pathology, oncology, and radiology",
                "Using CheXpert for medical image analysis"
            ],
            "required_readings": [
                "Irvin, J., Rajpurkar, P., Ko, M., et al. (2019). CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. AAAI Conference on Artificial Intelligence.",
                "Lu, M.Y., Chen, T.Y., Williamson, D.F.K. et al. AI-based pathology predicts origins for cancers of unknown primary. Nature 594, 106–110 (2021).",
                "Yala, A., Mikhael, P. G., Strand, F., Lin, G., Smith, K., Wan, Y. L., ... & Barzilay, R. (2021). Toward robust mammography-based models for breast cancer risk. Science Translational Medicine, 13(578), eaba4373."
            ]
        },
        {
            "week": 6,
            "title": "Medical Image Analysis III",
            "concepts": [
                "Biomedical image segmentation",
                "U-Net architecture",
                "Model performance across diverse populations"
            ],
            "required_readings": [
                "Antonelli, M., Reinke, A., Bakas, S., Farahani, K., Kopp-Schneider, A., Landman, B. A., ... & Cardoso, M. J. (2022). The medical segmentation decathlon. Nature communications, 13(1), 4128.",
                "Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI."
            ]
        },
        {
            "week": 7,
            "title": "Trustworthy AI",
            "concepts": [
                "Explainability and interpretability in medical AI",
                "Feature importance and Shapley values",
                "Interpreting CNNs with heatmaps and other methods",
                "Discussion: Is explainability critical or overrated?"
            ],
            "required_readings": [
                "Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). 'Why should I trust you?' Explaining the predictions of any classifier. ACM SIGKDD.",
                "Ghassemi, M., Oakden-Rayner, L., & Beam, A. L. (2021). The false hope of current approaches to explainable AI in health care. The Lancet Digital Health, 3(11), e745-e750."
            ]
        },
        {
            "week": 8,
            "title": "Networks I",
            "concepts": [
                "Multidimensional analysis of complex data",
                "Introduction to Graph Neural Networks (GNNs)",
                "Subgraph neural networks and graph transformers",
                "Combining multiple data modalities with GNNs"
            ],
            "required_readings": [
                "Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.",
                "Hamilton, W., Ying, Z., & Leskovec, J. (2017). Representation learning on graphs: Methods and applications. IEEE Data Engineering Bulletin."
            ]
        },
        {
            "week": 9,
            "title": "Networks II",
            "concepts": [
                "Knowledge graphs",
                "Building multimodal knowledge graphs",
                "Structure-inducing pre-training",
                "Knowledge-guided deep learning and inductive biases"
            ],
            "required_readings": [
                "Wang, X., Zhang, X., Qi, G. J., et al. (2020). KGAT: Knowledge graph attention network for recommendation. KDD.",
                "Zhu, H., Liu, H., Shen, Y., et al. (2021). Learning to pre-train graph neural networks. arXiv preprint arXiv:2103.11259."
            ]
        },
        {
            "week": 10,
            "title": "Networks III",
            "concepts": [
                "AI for protein structure prediction",
                "Drug discovery and therapeutic science",
                "Antibody design",
                "Structure- and sequence-based co-design",
                "Deep learning on genomic data"
            ],
            "required_readings": [
                "Jumper, J., Evans, R., Pritzel, A., et al. (2021). Highly accurate protein structure prediction with AlphaFold. Nature, 596(7873), 583-589.",
                "Stokes, J. M., Yang, K., Swanson, K., et al. (2020). A deep learning approach to antibiotic discovery. Cell, 180(4), 688-702."
            ]
        },
        {
            "week": 11,
            "title": "Multimodal AI",
            "concepts": [
                "Addressing label scarcity in medical data",
                "Semi-supervised and self-supervised learning",
                "Combining image and text modalities in AI (ConVIRT, CLIP)",
                "Exploring models like CheXzero and DALL-E"
            ],
            "required_readings": [
                "Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision. ICML.",
                "Zhang, Z., Hu, H., Xu, Z., et al. (2020). ConVIRT: Contrastive learning of medical visual representations. Medical Image Computing and Computer-Assisted Intervention (MICCAI)."
            ]
        },
        {
            "week": 12,
            "title": "Time Series & Sensors in Healthcare",
            "concepts": [
                "Digital biomarkers and disease progression tracking",
                "Patient/disease progression modeling using transformers",
                "In-home health and disease monitoring systems",
                "Intelligent and accessible AI systems for healthcare delivery"
            ],
            "required_readings": [
                "Solares, J. R., Raimondi, F., Zhu, Y., et al. (2020). Deep learning for electronic health records: A comparative review of multiple deep neural architectures. Journal of Biomedical Informatics, 101, 103337.",
                "Shickel, B., Tighe, P. J., Bihorac, A., & Rashidi, P. (2018). Deep EHR: A survey of recent advances in deep learning techniques for electronic health record analysis. IEEE Journal of Biomedical and Health Informatics, 22(5), 1589-1604."
            ]
        },
        {
            "week": 13,
            "title": "Ethical & Legal Considerations in AI for Medicine",
            "concepts": [
                "Regulation of AI algorithms and devices in healthcare",
                "FDA oversight and liability concerns",
                "Prospective clinical trials for AI systems and AI-augmented devices"
            ],
            "required_readings": [
                "McCradden, M. D., Stephenson, E. A., & Anderson, J. A. (2020). Regulatory frameworks for development and evaluation of AI-based diagnostic aids. Clinical Radiology, 75(9), 708-714.",
                "Price, W. N., & Cohen, I. G. (2019). Privacy in the age of medical big data. Nature Medicine, 25(1), 37-43."
            ]
        }
    ]
}
