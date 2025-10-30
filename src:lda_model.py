# src/lda_model.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from src.text_processor import ImprovedMedicalProcessor
from src.config import (
    N_TOPICS, RANDOM_STATE, MAX_DF, MIN_DF, MAX_FEATURES, NGRAM_RANGE,
    MAX_ITER_LDA, LEARNING_METHOD, COLUMN_COMPLAINT, COLUMN_HISTORY,
    COLUMN_DIAGNOSIS, COLUMN_TRIAGE, CONFIDENCE_THRESHOLD
)

logger = logging.getLogger(__name__)

class JournalReadyLDA:
    """
    An end-to-end pipeline for LDA topic modeling, optimized for
    medical texts. Includes preprocessing, training, and topic validation.
    """

    def __init__(self, n_topics=N_TOPICS, random_state=RANDOM_STATE):
        self.n_topics = n_topics
        self.random_state = random_state
        self.medical_processor = ImprovedMedicalProcessor()

        # Enhanced vectorizer with stricter parameters
        self.vectorizer = CountVectorizer(
            max_df=MAX_DF,
            min_df=MIN_DF,
            max_features=MAX_FEATURES,
            token_pattern=r'\b[a-zA-ZçğıöşüÇĞIİÖŞÜ_]{4,}\b', # Min 4 chars
            ngram_range=NGRAM_RANGE,
            lowercase=True
        )

        # IMPROVED LDA parameters
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            learning_method=LEARNING_METHOD,
            max_iter=MAX_ITER_LDA,
            n_jobs=1,
            verbose=0
        )
        
        self.doc_topic_matrix = None
        self.doc_term_matrix = None
        self.feature_names = None

    def preprocess_texts_with_enhancement(self, df):
        """
        Applies preprocessing and smart text enhancement to the DataFrame.
        Short or empty texts are enriched using Diagnosis or Triage info.
        """
        print("🔍 IMPROVED TEXT PREPROCESSING")
        print("-" * 50)

        # Combine primary text sources
        df['combined_text'] = (
            df[COLUMN_COMPLAINT].fillna('').astype(str) + ' ' +
            df[COLUMN_HISTORY].fillna('').astype(str)
        ).str.strip()

        processed_texts = []
        stats = {'original_valid': 0, 'diagnostic_enhanced': 0, 'minimal_enhanced': 0}

        print("⚡ Processing with improved smart enhancement...")

        for idx, row in df.iterrows():
            original_text = self.medical_processor.preprocess_medical_text(row['combined_text'])

            if len(original_text) > 15: # Stricter threshold
                processed_texts.append(original_text)
                stats['original_valid'] += 1
            else:
                # Enhance text strategy
                enhanced_text = original_text

                # Strategy 1: Add diagnostic information
                if COLUMN_DIAGNOSIS in df.columns and not pd.isna(row[COLUMN_DIAGNOSIS]):
                    diag_text = self.medical_processor.preprocess_medical_text(str(row[COLUMN_DIAGNOSIS]))
                    if diag_text:
                        enhanced_text += f" {diag_text}"
                        stats['diagnostic_enhanced'] += 1

                # Strategy 2: Minimal fallback
                if len(enhanced_text.strip()) <= 10:
                    enhanced_text = "acil_servis_basvuru medikal_muayene"
                    stats['minimal_enhanced'] += 1

                processed_texts.append(enhanced_text.strip())
            
            if idx % 100000 == 0 and idx > 0:
                print(f"  Processed: {idx:,} / {len(df):,}")

        df['processed_text'] = processed_texts
        
        print(f"\n📊 IMPROVED TEXT PROCESSING RESULTS:")
        print(f"   ✅ Original adequate: {stats['original_valid']:,} ({stats['original_valid']/len(df)*100:.1f}%)")
        print(f"   🔧 Diagnostic enhanced: {stats['diagnostic_enhanced']:,}")
        print(f"   🎯 ALL {len(df):,} PATIENTS RETAINED - NO EXCLUSIONS")

        return df

    def fit_and_transform_improved(self, texts):
        """
        Trains the LDA model on the preprocessed texts.
        """
        print(f"\n🔬 IMPROVED LDA TRAINING ({self.n_topics} topics)")
        print("-" * 50)
        start_time = datetime.now()

        # Vectorization
        print("📊 Creating document-term matrix...")
        self.doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"  ✅ Vocabulary: {len(self.feature_names):,} terms")
        print(f"  📊 Matrix shape: {self.doc_term_matrix.shape}")

        # LDA training
        lda_start = datetime.now()
        print("🧠 Training LDA model...")
        
        try:
            self.doc_topic_matrix = self.lda_model.fit_transform(self.doc_term_matrix)
            
            lda_time = (datetime.now() - lda_start).total_seconds()
            print(f"  ✅ LDA training completed in {lda_time:.1f}s")
            print(f"  📈 Converged in {self.lda_model.n_iter_} iterations")
            print(f"  📊 Log-likelihood: {self.lda_model.score(self.doc_term_matrix):.2f}")

        except Exception as e:
            logger.error(f"Improved LDA training failed: {e}")
            raise
            
        return self.doc_topic_matrix

    def get_improved_topics_with_validation(self, n_words=20):
        """
        Extracts topics, applies medical validation, and assesses quality.
        """
        topics = []
        print(f"\n🎯 IMPROVED TOPIC ANALYSIS")
        print("-" * 50)

        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[::-1][:n_words]
            top_words = [self.feature_names[i] for i in top_indices]
            top_weights = topic[top_indices]

            # Improved medical categorization
            medical_category, confidence, _ = \
                self.medical_processor.enhanced_medical_categorization(
                    top_words[:15], top_weights[:15], confidence_threshold=CONFIDENCE_THRESHOLD
                )

            # Enhanced quality assessment
            topic_quality = self._assess_improved_topic_quality(top_words, top_weights)

            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': top_weights,
                'medical_category': medical_category,
                'categorization_confidence': confidence,
                'quality_score': topic_quality,
                'interpretation_confidence': self._improved_confidence(confidence, topic_quality)
            })

        return topics

    def _assess_improved_topic_quality(self, words, weights):
        """
        Calculates a quality score based on term relevance and concentration.
        """
        # Weight concentration
        weight_concentration = np.sum(weights[:5]) / (np.sum(weights) + 1e-10)

        # Medical relevance (simple check for now)
        medical_terms = 0
        for word in words[:10]:
            if '_' in word: # Assumes compound terms are medically relevant
                medical_terms += 1
        medical_relevance = medical_terms / min(len(words), 10)

        # Word quality (average length)
        avg_word_length = np.mean([len(w) for w in words[:10]])
        length_quality = min(avg_word_length / 8, 1.0) # Target avg length of 8

        # Combined quality score
        quality_score = (weight_concentration * 0.4 +
                        medical_relevance * 0.5 +
                        length_quality * 0.1)
        return max(quality_score, 0.0)

    def _improved_confidence(self, categorization_conf, quality):
        """
        Combines categorization confidence and quality into an overall score.
        """
        overall = (categorization_conf * 0.6 + quality * 0.4)
        if overall > 0.8: return 'very_high'
        if overall > 0.6: return 'high'
        if overall > 0.4: return 'medium'
        return 'low'