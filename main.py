from scripts.pipeline import singleton_pipeline

if __name__ == '__main__':
    docs_path = ".\\documents\\labor_law"
    singleton_pipeline.parse_documents(docs_path)
    singleton_pipeline.vectorize_documents()

    collection = 'labor_law'
    query = '工作时间和休息休假是如何保障的'
    result = singleton_pipeline.search(collection, query, count=10, top_k=2)
    print('--> ', result)
