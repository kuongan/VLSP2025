import re
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator


class TextPreprocessor:
    def __init__(self,  translate_src="auto", translate_target="en"):
        """Khởi tạo với stopwords và ngôn ngữ đích cho translate."""
        self.translate_src = translate_src
        self.translate_target = translate_target

    def normalize_whitespace(self, text: str) -> str:
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        return '\n'.join(lines)

    def to_lowercase(self, text: str) -> str:
        return text.lower()

    def remove_html_tags(self, text: str) -> str:
        return BeautifulSoup(text, "html.parser").get_text()

    def translate_text(self, text: str) -> str:
        """Dịch văn bản sang ngôn ngữ đích."""
        try:
            translated = GoogleTranslator(source=self.translate_src, target=self.translate_target).translate(text)
            return translated
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return text

    def preprocess_plain_text(self, text: str) -> str:
        """Pipeline: normalize -> lowercase -> remove HTML -> remove special chars -> remove numbers -> stopwords -> translate."""
        text = self.normalize_whitespace(text)
        text = self.to_lowercase(text)
        text = self.remove_html_tags(text)
        text = self.translate_text(text)
        return text


# if __name__ == "__main__":
#    
#     raw = """
#     <html><body><h1>Xin chào Thế Giới!</h1><p>Đây là 1 đoạn văn bản cần dịch sang tiếng Anh.</p></body></html>
#     """
#     tp = TextPreprocessor(translate_target="en")

#     processed = tp.preprocess_plain_text(raw)
#     print(f"✅ Cleaned & Translated text: {processed}")
