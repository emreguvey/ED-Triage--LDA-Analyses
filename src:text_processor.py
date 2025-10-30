# src/text_processor.py
import pandas as pd
import re
import logging
from src.config import NLTK_AVAILABLE, TURKISH_STOPWORDS

logger = logging.getLogger(__name__)

class ImprovedMedicalProcessor:
    """
    Improved medical text processor with enhanced artifact removal,
    term standardization, and medical categorization.
    """

    def __init__(self):
        # EXPANDED stop words
        self.stop_words = {
            # Basic Turkish
            've', 'ile', 'bu', 'şu', 'o', 'bir', 'için', 'de', 'da', 'mi', 'mı',
            'var', 'yok', 'olan', 'oldu', 'geldi', 'gitti', 'ama', 'fakat',
            'veya', 'ya', 'hem', 'hiç', 'her', 'tüm', 'bütün', 'kendi',
            # Medical process terms (non-discriminatory)
            'hasta', 'şikayet', 'yakınma', 'tespit', 'edildi', 'yapılan',
            'muayene', 'inceleme', 'takip', 'kontrol', 'başvuru', 'getirilen',
            # Time/quantity terms
            'gün', 'saat', 'dakika', 'hafta', 'ay', 'yıl', 'önce', 'sonra',
            'çok', 'daha', 'en', 'az', 'biraz', 'oldukça', 'aşırı', 'fazla',
            # Generic qualifiers
            'gibi', 'kadar', 'beri', 'şekilde', 'biçimde', 'nedeni', 'sebebi',
            # EXPANDED: Problematic terms identified in analysis
            'olmuş', 'olmamiş', 'başlamış', 'başlayan', 'başlamiş', 'bugün',
            'gündür', 'kayeti', 'şikayetleri', 'hissi', 'bilinen', 'servise',
            'acil', 'getirilen', 'yapılan', 'edildi', 'tespit'
        }

        # Add NLTK Turkish stop words if available
        if NLTK_AVAILABLE:
            self.stop_words.update(TURKISH_STOPWORDS)

        # ENHANCED MEDICAL CATEGORIES with stricter definitions
        self.medical_categories = {
            'gastrointestinal': {
                'primary': ['karın_ağrısı', 'mide_ağrısı', 'bulantı', 'kusma', 'ishal', 'kabızlık'],
                'secondary': ['karın', 'mide', 'bağırsak', 'safra', 'karaciğer', 'endoskopi'],
                'symptoms': ['bulantı', 'kusma', 'ishal', 'karın_ağrısı', 'mide_ağrısı']
            },
            'respiratory': {
                'primary': ['öksürük', 'balgam', 'nefes_darlığı', 'boğaz_ağrısı', 'astım'],
                'secondary': ['boğaz', 'burun', 'akciğer', 'bronş', 'dispne', 'takipne'],
                'symptoms': ['öksürük', 'balgam', 'nefes_darlığı', 'boğaz_ağrısı']
            },
            'cardiovascular': {
                'primary': ['göğüs_ağrısı', 'çarpıntı', 'kalp_ağrısı', 'nefes_darlığı'],
                'secondary': ['kalp', 'çarpıntı', 'hipertansiyon', 'ritim', 'nabız', 'ekokardiyografi'],
                'symptoms': ['göğüs_ağrısı', 'kalp_ağrısı', 'çarpıntı']
            },
            'neurological': {
                'primary': ['baş_ağrısı', 'migren', 'bilinç_bulanıklığı', 'nöbet'],
                'secondary': ['baş', 'beyin', 'sinir', 'epilepsi', 'felç', 'tomografi'],
                'symptoms': ['baş_ağrısı', 'migren', 'baş_dönmesi']
            },
            'musculoskeletal': {
                'primary': ['bel_ağrısı', 'boyun_ağrısı', 'eklem_ağrısı', 'kas_ağrısı'],
                'secondary': ['bel', 'boyun', 'eklem', 'kas', 'kemik', 'ortopedik'],
                'symptoms': ['bel_ağrısı', 'boyun_ağrısı', 'eklem_ağrısı']
            },
            'trauma': {
                'primary': ['kaza', 'düşme', 'darbe', 'yaralanma', 'travma'],
                'secondary': ['kırık', 'kanama', 'çarpma', 'ezilme', 'yaralama'],
                'symptoms': ['ağrı', 'şişlik', 'kanama', 'hareket_kısıtlılığı']
            }
        }

        # Enhanced medical standardization rules
        self.medical_standardization = {
            'nod': 'nefes_darlığı', 'go': 'göğüs_ağrısı', 'ba': 'baş_ağrısı',
            'nös': 'nefes_darlığı', 'nefes_darlığıliği': 'nefes_darlığı',
            'ağrısı': 'ağrı', 'ağrisi': 'ağrı', 'agri': 'ağrı', 'ağri': 'ağrı',
            'bulanti': 'bulantı', 'karin': 'karın', 'shal': 'ishal',
            'ekg': 'elektrokardiyogram', 'bt': 'bilgisayarlı_tomografi',
            'mr': 'manyetik_rezonans'
        }

    def preprocess_medical_text(self, text):
        """
        Applies the full preprocessing pipeline to a single medical text.
        """
        if pd.isna(text) or text == '':
            return ""

        text = str(text).lower()

        # ENHANCED artifact removal
        text = re.sub(r'_[a-z]+_', ' ', text)  # Remove _xd_ type artifacts
        text = re.sub(r'[^\w\s]', ' ', text)   # Remove punctuation
        text = re.sub(r'\d+', '', text)        # Remove numbers

        # Enhanced standardization
        for abbr, standard in self.medical_standardization.items():
            pattern = r'\b' + abbr + r'\b'
            text = re.sub(pattern, standard, text)

        # Enhanced compound medical terms
        text = re.sub(r'baş\s*ağr[ıi]', 'baş_ağrısı', text)
        text = re.sub(r'göğüs\s*ağr[ıi]', 'göğüs_ağrısı', text)
        text = re.sub(r'nefes\s*dar', 'nefes_darlığı', text)
        text = re.sub(r'karın\s*ağr[ıi]', 'karın_ağrısı', text)
        text = re.sub(r'bel\s*ağr[ıi]', 'bel_ağrısı', text)
        text = re.sub(r'boyun\s*ağr[ıi]', 'boyun_ağrısı', text)

        # Clean and normalize
        text = re.sub(r'\s+', ' ', text)       # Normalize whitespace

        # Enhanced tokenization and filtering
        words = text.split()
        filtered_words = []

        for word in words:
            if (len(word) > 3 and len(word) < 25 and
                word not in self.stop_words and
                not word.isdigit()):
                filtered_words.append(word)

        return ' '.join(filtered_words)

    def enhanced_medical_categorization(self, top_words, weights=None, confidence_threshold=0.8):
        """
        Categorizes a topic based on its top words using stricter rules.
        """
        word_string = ' '.join(top_words).lower()
        category_scores = {}
        category_details = {}

        for category, word_groups in self.medical_categories.items():
            score = 0
            matched_words = []

            for word in word_groups['primary']:
                if word in word_string:
                    score += 5
                    matched_words.append((word, 'primary'))
            for word in word_groups['secondary']:
                if word in word_string:
                    score += 2
                    matched_words.append((word, 'secondary'))
            for word in word_groups.get('symptoms', []):
                if word in word_string:
                    score += 3
                    matched_words.append((word, 'symptom'))

            category_scores[category] = score
            category_details[category] = { 'score': score, 'matched_words': matched_words }

        if not category_scores or max(category_scores.values()) == 0:
            return 'general_medical', 0.0, {}

        best_category = max(category_scores.keys(), key=category_scores.get)
        max_score = category_scores[best_category]

        # Calculate confidence
        sorted_scores = sorted(category_scores.values(), reverse=True)
        confidence = (sorted_scores[0] - sorted_scores[1]) / (sorted_scores[0] + 1e-10) if len(sorted_scores) > 1 else 1.0
        
        # Apply STRICTER confidence threshold
        if confidence < confidence_threshold or max_score < 8:
            return 'general_medical', confidence, category_details

        return best_category, confidence, category_details