'''
The scourse time table for AIM2 course. This course can be found at:
https://zitniklab.hms.harvard.edu/AIM2/
'''


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
                "Clinical Note summarization",
                "Clinical trial matching"
            ],
            "required_readings": [
                "Singhal, et al. (2023). Large language models encode clinical "
                "knowledge.",
                "Himmelstein, G., et al. (2022). Examination of stigmatizing "
                "language in the electronic health record."
            ]
        },
        {
            "week": 2,
            "title": "NLP II: Embeddings & Transformers",
            "concepts": [
                "Embeddings and their role in NLP",
                "Transformers and BERT Hugging Face library for NLP "
                "applications",
                "Clinical BERT and RNNs",
                "Stack-encoder and Stack-decoder architectures",
                "De-identification methods",
                "LLM-based medical question-answering"
            ],
            "required_readings": [
                "Devlin, J., et al. (2019). BERT: Pre-training of deep "
                "bidirectional transformers for language understanding.",
                "Lee, J., et al. (2020). BioBERT: a pre-trained biomedical "
                "language representation model. "
            ]
        },
        {
            "week": 3,
            "title": "Generative AI",
            "concepts": [
                "Variational Autoencoders (VAEs)",
                "Generative Adversarial Networks (GANs)",
                "GenAI Fundamentals (training, optimization and Reinforcement Learning from Human Feedback)",
                "LArge Language Models and Multimodal Large Language Models",
                "Grounding and Retrieval Augmented Generators",
                "Healthcare applications of generative artificial intelligence",
                "Synthetic data generation",
                "Data privacy concerns"
            ],
            "required_readings": [
                "Goodfellow, I., et al. (2014). Generative adversarial nets.",
                "Gulrajani, I., et al. (2020). In search of lost domain "
                "generalization."
            ]
        },
        {
            "week": 4,
            "title": "Agentic AI",
            "concepts": [
                "Designing LLM-driven agents to answer complex clinical "
                "queries with evidence-backed reasoning",
                "Strategies to evaluate accuracy, robustness, "
                "and interpretability in high-stakes medical contexts",
                "Case studies of LLM-based agents in clinical decision-making, "
                "drug discovery, and patient triage",
                "Emerging trends, such as real-time conversational agents, collaborative "
                "multi-agent systems, and autonomous discovery"
            ],
            "required_readings": [
                "Tu, T., et al. (2024). Towards conversational diagnostic AI. ",
                "Boiko, D.A., et al. (2023). Autonomous chemical research with "
                "large language models. "
            ]
        },
        {
            "week": 5,
            "title": "Medical Image Analysis I",
            "concepts": [
                "Understanding medical imaging modalities",
                "Understand the various types of medical imaging (radiology, oncology, "
                "pathology, and other imaging modalities).",
                "Learn the basic tasks in AI for medical imaging: classification, regression, and segmentation.",
                "Explore how AI is applied in different medical imaging contexts.",
                "Understand convolutional neural networks (CNNs) and their role in medical "
                "imaging.",
                "Learn segmentation techniques, focusing on the U-Net architecture.",
                "Apply CNNs to biomedical image segmentation tasks, including preprocessing "
                "and evaluation."
            ],
            "required_readings": [
                "Litjens, G., et al. (2017). A survey on deep learning in medical image analysis.",
                "Esteva, A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. ",
                "Antonelli, M., et al. (2022). The medical segmentation "
                "decathlon.",
                "Ronneberger, O., et al. (2015). U-Net: Convolutional networks "
                "for biomedical image segmentation."
            ]
        },
        {
            "week": 6,
            "title": "Medical Image Analysis II",
            "concepts": [
                "Explore advanced applications of AI in medical imaging, "
                "with a focus on generalist medical AI models.",
                "Understand the development and validation of medical imaging interpretation "
                "models.",
                "Discuss best practices for evaluating medical imaging AI models, "
                "with emphasis on robustness and performance across diverse populations.",

            ],
            "required_readings": [
                "Moor, M., et al. (2023) Foundation models for generalist "
                "medical artificial intelligence.",
                "Tiu, E., et al. (2022) Expert-level detection of pathologies "
                "from unannotated chest X-ray images via self-supervised learning. ",
                "Zhou, H. Y., et al. (2024). A Generalist Learner for "
                "Multifaceted Medical Image Interpretation. "

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
                "Ribeiro, M. T., et al. (2016). 'Why should I trust you?' "
                "Explaining the predictions of any classifier.",
                "Lundberg, S.M., et al. (2018). Explainable machine-learning "
                "predictions for the prevention of hypoxaemia during surgery.",
                "Ghassemi, M., et al. (2021). The false hope of current "
                "approaches to explainable AI in health care."
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
                "Grover, A. et al. (2016). node2vec: Scalable feature "
                "learning for networks.",
                "Li, M.M., eta l. (2022). Graph representation learning in "
                "biomedicine and healthcare.",
                "Ruiz, C., et al.  (2021). Identification of disease treatment "
                "mechanisms through the multiscale interactome. "
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
                "Nelson, C.A., et al. (2019). Integrating biomedical research "
                "and electronic health records to create knowledge-based biologically meaningful machine-readable embeddings. ",
                "Huang, K., et al. (2024). A foundation model for "
                "clinician-centered drug repurposing."
            ]
        },
        {
            "week": 10,
            "title": "Molecular AI",
            "concepts": [
                "AI for protein structure prediction",
                "Drug discovery and therapeutic science",
                "Antibody design",
                "Structure- and sequence-based co-design",
                "Deep learning on genomic data"
            ],
            "required_readings": [
                "Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold.",
                "Stokes, J. M., et al. (2020). A deep learning approach to "
                "antibiotic discovery."
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
                "Radford, A., et al. (2021). Learning transferable visual models from natural language supervision.",
                "Zhang, Z., et al. (2020). ConVIRT: Contrastive learning of "
                "medical visual representations."
            ]
        },
        {
            "week": 12,
            "title": "Ethical & Legal Considerations in AI for Medicine",
            "concepts": [
                "Regulation of AI algorithms and devices in healthcare",
                "FDA oversight and liability concerns",
                "Prospective clinical trials for AI systems and AI-augmented devices"
            ],
            "required_readings": [
                "Gerke, S., et al. (2020). Ethical and legal aspects of "
                "ambient intelligence in hospitals. ",
                "Price, W. N., et al. (2019). Privacy in the age of medical "
                "big data.",
                "Babic, B., et al. (2019). Algorithms on regulatory lockdown "
                "in medicine. ",
                "Price, W.N., et al. (2019). Potential liability for "
                "physicians using artificial intelligence."
            ]
        },
        {
            "week": 13,
            "title": "Time Series & Sensors in Healthcare",
            "concepts": [
                "Digital biomarkers and disease progression tracking",
                "Patient/disease progression modeling using transformers",
                "In-home health and disease monitoring systems",
                "Intelligent and accessible AI systems for healthcare delivery"
            ],
            "required_readings": [
                "Yang, Y., et al. (2022). Artificial intelligence-enabled "
                "detection and assessment of Parkinson’s disease using nocturnal breathing signals. ",
                "Cohen, N.M., et al. (2024). Longitudinal machine learning "
                "uncouples healthy aging factors from chronic disease risks. "
            ]
        }

    ]
}
