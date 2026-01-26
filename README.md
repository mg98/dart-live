# DART for Deployment

In this repository, I plan to push files related to the planned deployment of online learning-to-rank within Tribler.

In the meantime, you can enjoy an interactive terminal application I built that lets you explore the effects of the PDGD ranker trained on our dataset.


> [!IMPORTANT]
> Run Tribler in the background, as our process will try to connect to its REST API.

```
python tribler_search.py --ltr --remote
```

- Remove `--remote` if you want to skip P2P search and only use local documents.
- Remove `--ltr` if you want to disable reranking; this will render results with Tribler's default ranking.
