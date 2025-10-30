# main.py
import os
import gc
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import entropy

from src.config import (
    OUTPUT_DIR, N_TOPICS, RANDOM_STATE, COLUMN_GROUP
)
from src.data_loader import load_all_data
from src.lda_model import JournalReadyLDA
from src.visualization import IndividualChartExporter

# Set up enhanced logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_journal_ready_analysis():
    """
    Main function to run the complete, improved journal-ready medical LDA analysis.
    Orchestrates data loading, preprocessing, modeling, feature engineering,
    and visualization.
    """
    start_time = datetime.now()
    
    print("🏥 IMPROVED JOURNAL-READY MEDICAL LDA ANALYSIS")
    print("="*70)
    logger.info("Starting journal-ready analysis")

    try:
        # --- 1. Load Data ---
        print("\n📁 STEP 1: DATA LOADING")
        df = load_all_data()
        if df is None:
            print("❌ Data loading failed. Exiting.")
            return

        load_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Data loaded in {load_time:.1f}s. Total rows: {len(df):,}")
        print(f"   Found {df[COLUMN_GROUP].nunique()} data sources (groups).")

        # --- 2. Create Output Directory ---
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"📂 Output directory ready: {OUTPUT_DIR}")

        # --- 3. Text Preprocessing ---
        print("\n🔤 STEP 2: TEXT PREPROCESSING")
        lda_pipeline = JournalReadyLDA(n_topics=N_TOPICS, random_state=RANDOM_STATE)
        df = lda_pipeline.preprocess_texts_with_enhancement(df)
        
        preprocess_time = (datetime.now() - start_time).total_seconds() - load_time
        print(f"✅ Preprocessing completed in {preprocess_time:.1f}s")

        # --- 4. LDA Modeling ---
        print("\n🧠 STEP 3: LDA MODELING")
        doc_topic_matrix = lda_pipeline.fit_and_transform_improved(df['processed_text'].tolist())
        topics_data = lda_pipeline.get_improved_topics_with_validation()
        
        lda_time = (datetime.now() - start_time).total_seconds() - preprocess_time - load_time
        print(f"✅ LDA modeling completed in {lda_time:.1f}s")
        
        print("\n🎯 IMPROVED MEDICAL TOPICS:")
        print("-" * 40)
        for topic in topics_data:
            category = topic['medical_category'].replace('_', ' ').title()
            print(f"Topic {topic['topic_id']}: {category}")
            print(f"  Quality: {topic['quality_score']:.3f} | Confidence: {topic['categorization_confidence']:.3f}")
            print(f"  Overall: {topic['interpretation_confidence']}")
            print(f"  Top words: {', '.join(topic['words'][:8])}\n")
            
        # --- 5. Feature Engineering ---
        print("\n⚙️ STEP 4: ENHANCED FEATURE ENGINEERING")
        
        for i in range(lda_pipeline.n_topics):
            df[f'topic_{i}_prob'] = doc_topic_matrix[:, i]
        
        df['dominant_topic'] = np.argmax(doc_topic_matrix, axis=1)
        df['max_topic_prob'] = np.max(doc_topic_matrix, axis=1)
        df['topic_entropy'] = np.apply_along_axis(entropy, 1, doc_topic_matrix)

        # Map dominant topic to its medical category
        topic_to_category = {t['topic_id']: t['medical_category'] for t in topics_data}
        df['dominant_topic_medical_category'] = df['dominant_topic'].map(topic_to_category).fillna('general_medical')
        
        print(f"✅ Generated {lda_pipeline.n_topics + 3} new features.")

        # --- 6. Individual Chart Export ---
        print("\n📊 STEP 5: INDIVIDUAL CHART EXPORT")
        
        chart_exporter = IndividualChartExporter(OUTPUT_DIR, df, doc_topic_matrix, topics_data)
        chart_count = chart_exporter.export_all_individual_charts()
        
        # --- 7. Enhanced Results Export ---
        print("\n💾 STEP 6: ENHANCED EXCEL EXPORT")
        
        export_count = 0
        for group_name in df[COLUMN_GROUP].unique():
            group_df = df[df[COLUMN_GROUP] == group_name].copy()
            
            # Drop processing columns
            cols_to_drop = ['combined_text', 'processed_text']
            group_df = group_df.drop([col for col in cols_to_drop if col in group_df.columns], axis=1)
            
            filename = f"{group_name}_IMPROVED_JOURNAL_READY_LDA_FEATURES.xlsx"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            try:
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    # Main dataset
                    group_df.to_excel(writer, sheet_name='Improved_Features', index=False)
                    
                    # Topic descriptions
                    topic_descriptions_data = []
                    for t in topics_data:
                        prob_col = f"topic_{t['topic_id']}_prob"
                        prevalence = group_df[prob_col].mean() * 100 if prob_col in group_df else 0
                        
                        topic_descriptions_data.append({
                            'Topic_ID': t['topic_id'],
                            'Medical_Category': t['medical_category'].replace('_', ' ').title(),
                            'Quality_Score': f"{t['quality_score']:.3f}",
                            'Medical_Confidence': f"{t['categorization_confidence']:.3f}",
                            'Overall_Confidence': t['interpretation_confidence'],
                            'Top_Words': ', '.join(t['words'][:10]),
                            'Prevalence_In_Group': f"{prevalence:.1f}%"
                        })
                    
                    topic_descriptions_df = pd.DataFrame(topic_descriptions_data)
                    topic_descriptions_df.to_excel(writer, sheet_name='Topic_Descriptions', index=False)
                
                export_count += 1
                file_size = os.path.getsize(filepath) / (1024**2)
                print(f"  ✅ {filename}: {len(group_df):,} records ({file_size:.1f}MB)")
            
            except Exception as e:
                logger.error(f"Failed to export Excel for {group_name}: {e}")
                print(f"  ❌ FAILED to export for {group_name}: {e}")

        # --- 8. Final Summary ---
        total_time_mins = (datetime.now() - start_time).total_seconds() / 60
        
        print("\n" + "="*70)
        print("🎉 ANALYSIS COMPLETED!")
        print("="*70)
        print(f"   📄 Total patients processed: {len(df):,}")
        print(f"   🏥 Data groups/hospitals: {df[COLUMN_GROUP].nunique()}")
        print(f"   🎯 Topics discovered: {lda_pipeline.n_topics}")
        print(f"   ⏱️ Total processing time: {total_time_mins:.1f} minutes")
        print(f"   📊 {export_count} feature sets exported.")
        print(f"   📈 {chart_count} individual charts exported.")
        print(f"   📂 Output location: {OUTPUT_DIR}")
        print("="*70)
        
        # Cleanup
        gc.collect()
        
        logger.info("Journal-ready analysis completed successfully")
        return df, lda_pipeline, doc_topic_matrix, topics_data

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_journal_ready_analysis()

    if results:
        print("\n✅ Analysis successful. Objects are available in memory if run interactively.")
    else:
        print("\n❌ Analysis failed. Please check error messages above.")