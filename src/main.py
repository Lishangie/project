import os
os.environ["USER_AGENT"] = "my-rag/1.0"

from .graph import build_rag_graph

def main():
    rag = build_rag_graph()
    print(rag.invoke("What is the capital of France?"))

if __name__ == "__main__":
    main()