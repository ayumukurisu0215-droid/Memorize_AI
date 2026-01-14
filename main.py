import os
import time

# --- 1. 必要なライブラリのインポート ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# --- 2. 設定 (APIキーをここに入力) ---

# --- 3. 記憶システムの構築 (ChromaDB + Gemini) ---

def setup_memory_ai():
    print("🧠 記憶システム(ChromaDB)を起動中...")

    # A. 埋め込みモデルの準備
    # 言葉を数値(ベクトル)に変換する翻訳機です
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # B. ベクトルデータベース(ChromaDB)の準備
    # "./chroma_memory_db" というフォルダに記憶を永続保存します
    vectorstore = Chroma(
        collection_name="chat_history",
        embedding_function=embeddings,
        persist_directory="./chroma_memory_db"  # ここにファイルとして保存される
    )

    # C. AIモデル(Gemini)の準備
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

    return vectorstore, llm

# --- 4. 会話の処理ロジック ---

def chat_with_memory(user_input, vectorstore, llm):
    # A. 過去の記憶を検索 (Retrieve)
    # ユーザーの発言に「意味が近い」過去ログを3件探してくる
    search_results = vectorstore.similarity_search(user_input, k=3)
    
    # 検索結果をテキストにまとめる
    context_text = "\n".join([doc.page_content for doc in search_results])
    
    if not context_text:
        context_text = "（過去の関連する会話はありません）"

    # B. プロンプトの作成
    # AIに「過去の記憶」と「今の発言」を同時に渡す
    template = """
    あなたは親しい友人AIです。以下の【過去の記憶】をヒントにして、ユーザーの質問に答えてください。
    
    【過去の記憶】
    {context}
    
    【ユーザーの今の発言】
    {input}
    
    回答:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    # C. AIからの回答を生成
    response = chain.invoke({"context": context_text, "input": user_input})
    ai_message = response.content

    # D. 今の会話を記憶に保存 (Save)
    # 会話の内容をChromaDBに追加する
    # "User: ... / AI: ..." という形式で保存しておくと文脈が分かりやすい
    memory_text = f"User: {user_input} / AI: {ai_message}"
    vectorstore.add_documents([Document(page_content=memory_text)])

    return ai_message, context_text

# --- 5. メイン実行ループ ---

if __name__ == "__main__":
    # システムの初期化
    vectorstore, llm = setup_memory_ai()
    print("🤖 準備完了！会話を始めましょう (終了するには 'exit' と入力)")
    print("-" * 50)

    while True:
        user_input = input("あなた: ")
        
        if user_input.lower() == "exit":
            print("またね！(記憶は保存されました)")
            break
            
        # AIと会話
        response, context = chat_with_memory(user_input, vectorstore, llm)
        
        print(f"AI: {response}")
        
        # デバッグ用: AIが何を思い出していたかを表示
        # print(f"\n[思い出していたこと]:\n{context}\n")
        print("-" * 50)