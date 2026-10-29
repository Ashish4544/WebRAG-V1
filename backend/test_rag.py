from rag import WebRAG


url = "https://www.digitalocean.com/community/tutorials/why-rag-systems-fail-in-production"

query = "How do latency and embedding drift affect RAG systems?"


rag = WebRAG()


results = rag.retrieve(
    url=url,
    query=query,
    k=3
)
text = rag.load_webpage(url)

print("\n" + "=" * 60)
print("SEARCHING EXTRACTED TEXT")
print("=" * 60)

# for keyword in ["Tilak", "Dube", "Tilak Varma", "Shivam Dube"]:
#     print(f"\n--- {keyword} ---")

#     index = text.lower().find(keyword.lower())

#     if index != -1:
#         print(text[max(0, index - 500):index + 1000])
#     else:
#         print("Not found")

print("\n" + "=" * 60)
print("RETRIEVED RESULTS")
print("=" * 60)


for rank, result in enumerate(results, start=1):

    print(f"\n--- Result {rank} ---")
    print("RRF score:", result["reranker_score"])
    #print("Source:", result["reranker_score"])
    print("\n", result["text"])