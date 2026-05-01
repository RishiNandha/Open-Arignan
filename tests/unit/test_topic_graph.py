from arignan.graph import TopicGraphEntry, build_topic_graph


def test_build_topic_graph_links_related_topics_with_confidence_and_type() -> None:
    graph = build_topic_graph(
        [
            TopicGraphEntry(
                topic_folder="skipgram",
                title="Skipgram Distributed Representations",
                locator="skipgram and word2vec training ideas",
                description="Distributed word representation training notes.",
                keywords=["skipgram", "word2vec", "negative sampling"],
                summary_excerpt="Focuses on skipgram objectives and word2vec style training.",
            ),
            TopicGraphEntry(
                topic_folder="word2vec-training",
                title="Training Skipgrams",
                locator="word2vec optimization and negative sampling",
                description="Optimization notes for word2vec.",
                keywords=["word2vec", "skipgram", "negative sampling"],
                summary_excerpt="Covers negative sampling and skipgram training details.",
            ),
            TopicGraphEntry(
                topic_folder="jepa",
                title="JEPA Notes",
                locator="joint embedding predictive architecture overview",
                description="Predictive representation learning.",
                keywords=["jepa", "predictive architecture"],
                summary_excerpt="Latent prediction for representations.",
            ),
        ]
    )

    assert graph["skipgram"]
    relation = graph["skipgram"][0]
    assert relation["topic_folder"] == "word2vec-training"
    assert relation["relation_type"] == "EXTRACTED"
    assert relation["confidence"] >= 0.5
    assert "word2vec" in relation["shared_terms"]
    assert graph["jepa"] == []
