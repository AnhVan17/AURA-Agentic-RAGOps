#!/usr/bin/env python3
import os
import sys

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def main():
    print("=" * 60)
    print("LangSmith + LangChain + Gemini Integration Test")
    print("=" * 60)
    
    # ===== BƯỚC 1: Khởi tạo LangSmith =====
    print("\n Bước 1: Khởi tạo LangSmith...")
    
    from ops.observability import init_langsmith
    
    success = init_langsmith(project_name="academic-rag-chatbot")
    
    if not success:
        print("\nLangSmith chưa được cấu hình đầy đủ.")
        print("Để bật tracing, thêm vào file .env:")
        print("  LANGCHAIN_API_KEY=your_langsmith_api_key")
        print("\nLấy key tại: https://smith.langchain.com/")
        print("\n(Vẫn tiếp tục test LLM...)")
    
    # ===== BƯỚC 2: Gọi LLM qua LangChain =====
    print("\nBước 2: Gọi Gemini qua LangChain...")
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        
        # Lấy API key
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            print("GOOGLE_API_KEY không tìm thấy trong .env!")
            return
        
        # Tạo LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.3,
        )
        
        # Test message
        print("   Đang gửi request...")
        response = llm.invoke([
            HumanMessage(content="RAG trong AI là gì? Trả lời ngắn gọn 2 câu.")
        ])
        
        print("\nLLM Response:")
        print("-" * 40)
        print(response.content)
        print("-" * 40)
        
        # ===== BƯỚC 3: Test với LCEL Chain =====
        print("\nBước 3: Test LCEL Chain...")
        
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là trợ lý học thuật. Trả lời ngắn gọn."),
            ("human", "{question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        result = chain.invoke({"question": "Hybrid search là gì?"})
        
        print("\nChain Response:")
        print("-" * 40)
        print(result)
        print("-" * 40)
        
        # ===== KẾT THÚC =====
        print("\n" + "=" * 60)
        if success:
            print("THÀNH CÔNG!")
            print("Kiểm tra trace tại: https://smith.langchain.com/")
            print("Project: academic-rag-chatbot")
        else:
            print("LLM hoạt động tốt!")
            print("Tracing chưa được bật (cần LANGCHAIN_API_KEY)")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Thiếu package: {e}")
        print("Chạy: pip install langchain-google-genai langchain-core langsmith")
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
