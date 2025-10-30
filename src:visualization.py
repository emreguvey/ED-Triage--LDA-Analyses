# src/visualization.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from src.config import (
    WORDCLOUD_AVAILABLE, CHART_DPI, COLUMN_GROUP, COLUMN_TRIAGE, WORDCLOUD_MAX_WORDS
)

if WORDCLOUD_AVAILABLE:
    from wordcloud import WordCloud

logger = logging.getLogger(__name__)

class IndividualChartExporter:
    """
    Handles the creation and export of all individual analysis charts
    in a publication-ready format.
    """

    def __init__(self, output_dir, df, doc_topic_matrix, topics_data, validation_results=None):
        self.output_dir = output_dir
        self.df = df
        self.doc_topic_matrix = doc_topic_matrix
        self.topics_data = topics_data
        self.validation_results = validation_results
        self.n_topics = len(topics_data)
        self.topic_labels = [
            f"T{t['topic_id']}: {t['medical_category'].replace('_', ' ').title()}" 
            for t in self.topics_data
        ]

        # Set publication-quality style
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
            'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
            'figure.titlesize': 16, 'figure.dpi': CHART_DPI
        })

    def export_all_individual_charts(self):
        """
        Main function to export all defined charts as individual PNG files.
        """
        print("📊 Exporting individual charts...")
        chart_count = 0
        
        # List of plotting functions to run
        plot_functions = [
            (self._plot_topic_prevalence, '01_topic_prevalence.png', (12, 8)),
            (self._plot_patient_assignments, '02_patient_assignments.png', (12, 8)),
            (self._plot_group_comparison, '03_group_comparison.png', (12, 8)),
            (self._plot_topic_quality_metrics, '04_topic_quality_metrics.png', (12, 8)),
            (self._plot_document_coverage, '05_document_coverage.png', (12, 8)),
            (self._plot_triage_correlation, '06_triage_correlation.png', (10, 8)),
            (self._plot_topic_confidence_levels, '07_topic_confidence_levels.png', (10, 8))
        ]

        for plot_func, filename, figsize in plot_functions:
            try:
                fig, ax = plt.subplots(figsize=figsize)
                plot_func(ax) # Call the plotting function
                fig.tight_layout()
                fig.savefig(os.path.join(self.output_dir, filename),
                           dpi=CHART_DPI, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                print(f"  ✅ {filename}")
                chart_count += 1
            except Exception as e:
                logger.warning(f"Failed to generate chart {filename}: {e}")
                plt.close(fig) # Ensure figure is closed on error

        # --- Individual Topic Word Clouds ---
        if WORDCLOUD_AVAILABLE:
            for topic_idx in range(self.n_topics):
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    self._plot_topic_wordcloud(ax, topic_idx)
                    filename = f'{chart_count+1:02d}_topic_{topic_idx}_wordcloud.png'
                    fig.tight_layout()
                    fig.savefig(os.path.join(self.output_dir, filename),
                               dpi=CHART_DPI, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                    print(f"  ✅ {filename}")
                    chart_count += 1
                except Exception as e:
                    logger.warning(f"Failed to generate wordcloud for Topic {topic_idx}: {e}")
                    plt.close(fig)
        else:
            print("  ⚠️ WordCloud not available - skipping word cloud charts.")

        print(f"\n📊 Total individual charts exported: {chart_count}")
        return chart_count

    def _plot_topic_prevalence(self, ax):
        """Plot topic prevalence with 95% CI"""
        topic_means = self.doc_topic_matrix.mean(axis=0)
        topic_stds = self.doc_topic_matrix.std(axis=0)
        topic_ci = 1.96 * topic_stds / np.sqrt(len(self.doc_topic_matrix))
        x = np.arange(self.n_topics)

        bars = ax.bar(x, topic_means, yerr=topic_ci, capsize=6,
                     color='steelblue', alpha=0.8, edgecolor='navy')

        ax.set_xticks(x)
        ax.set_xticklabels(self.topic_labels, rotation=45, ha='right')
        ax.set_ylabel('Average Topic Probability')
        ax.set_title('Topic Prevalence with 95% Confidence Intervals', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_patient_assignments(self, ax):
        """Plot patient assignments by dominant topic"""
        dominant_topics = np.argmax(self.doc_topic_matrix, axis=1)
        topic_counts = np.bincount(dominant_topics, minlength=self.n_topics)
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_topics))
        
        bars = ax.bar(range(self.n_topics), topic_counts, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(self.n_topics))
        ax.set_xticklabels(self.topic_labels, rotation=45, ha='right')
        ax.set_ylabel('Number of Patients')
        ax.set_title('Patient Assignments by Dominant Topic', fontweight='bold')
        for bar, count in zip(bars, topic_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(topic_counts)*0.01,
                   f'{count:,}', ha='center', va='bottom', fontweight='bold')

    def _plot_group_comparison(self, ax):
        """Plot topic prevalence comparison across data groups/hospitals"""
        groups = self.df[COLUMN_GROUP].unique()
        n_groups = len(groups)
        if n_groups <= 1:
            ax.text(0.5, 0.5, 'Only one data group found.\nComparison plot skipped.',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Group Comparison (Skipped)', fontweight='bold')
            return

        x = np.arange(self.n_topics)
        width = 0.8 / n_groups
        colors = plt.cm.Set2(np.linspace(0, 1, n_groups))

        for i, group in enumerate(groups):
            mask = self.df[COLUMN_GROUP] == group
            group_means = self.doc_topic_matrix[mask].mean(axis=0)
            offset = (i - (n_groups - 1) / 2) * width
            ax.bar(x + offset, group_means, width, label=f"{group} (n={mask.sum():,})",
                   color=colors[i], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(self.topic_labels, rotation=45, ha='right')
        ax.set_ylabel('Average Topic Probability')
        ax.set_title('Data Group/Hospital Topic Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_topic_quality_metrics(self, ax):
        """Plot topic quality and medical confidence scores"""
        quality_scores = [t.get('quality_score', 0) for t in self.topics_data]
        confidence_scores = [t.get('categorization_confidence', 0) for t in self.topics_data]
        x = np.arange(self.n_topics)
        width = 0.35

        ax.bar(x - width/2, quality_scores, width, label='Quality Score', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, confidence_scores, width, label='Medical Confidence', color='lightcoral', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f'Topic {i}' for i in range(self.n_topics)])
        ax.set_ylabel('Score')
        ax.set_title('Topic Quality and Medical Confidence Scores', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_document_coverage(self, ax):
        """Plot distribution of max topic probabilities per document"""
        max_probs = np.max(self.doc_topic_matrix, axis=1)
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ['<0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '>0.8']
        
        categories = pd.cut(max_probs, bins=bins, labels=labels, right=True)
        category_counts = categories.value_counts().sort_index()
        percentages = category_counts / len(max_probs) * 100

        bars = ax.bar(category_counts.index, percentages, color='forestgreen', alpha=0.8, edgecolor='black')
        ax.set_ylabel('Percentage of Patients (%)')
        ax.set_xlabel('Maximum Topic Probability')
        ax.set_title('Document Topic Assignment Quality', fontweight='bold')
        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    def _plot_triage_correlation(self, ax):
        """Plot heatmap of average topic probability by triage level"""
        if COLUMN_TRIAGE not in self.df.columns:
            ax.text(0.5, 0.5, 'Triage column not found.\nCorrelation plot skipped.',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Triage Correlation (Skipped)', fontweight='bold')
            return

        triage_categories = sorted(self.df[COLUMN_TRIAGE].dropna().unique())
        correlation_matrix = np.zeros((len(triage_categories), self.n_topics))

        for i, triage in enumerate(triage_categories):
            mask = self.df[COLUMN_TRIAGE] == triage
            if mask.sum() > 0:
                correlation_matrix[i] = self.doc_topic_matrix[mask].mean(axis=0)

        im = ax.imshow(correlation_matrix, cmap='YlOrRd', aspect='auto')
        for i in range(len(triage_categories)):
            for j in range(self.n_topics):
                ax.text(j, i, f'{correlation_matrix[i, j]:.2f}', ha="center", va="center", color="black")

        ax.set_xticks(range(self.n_topics))
        ax.set_xticklabels([f'T{i}' for i in range(self.n_topics)])
        ax.set_yticks(range(len(triage_categories)))
        ax.set_yticklabels(triage_categories)
        ax.set_title('Average Topic Probability by Triage Level', fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Average Probability')

    def _plot_topic_confidence_levels(self, ax):
        """Plot bar chart of topic interpretation confidence levels"""
        levels = [t.get('interpretation_confidence', 'unknown') for t in self.topics_data]
        level_counts = pd.Series(levels).value_counts().reindex(
            ['very_high', 'high', 'medium', 'low', 'unknown']).dropna()
        
        colors = {'very_high': 'green', 'high': 'limegreen', 'medium': 'orange', 'low': 'red', 'unknown': 'grey'}
        bar_colors = [colors.get(level, 'grey') for level in level_counts.index]
        
        bars = ax.bar(level_counts.index, level_counts.values, color=bar_colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Number of Topics')
        ax.set_title('Topic Interpretation Confidence Levels', fontweight='bold')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f"{int(bar.get_height())}", ha='center', va='bottom', fontweight='bold')

    def _plot_topic_wordcloud(self, ax, topic_idx):
        """Plot word cloud for a specific topic"""
        if not WORDCLOUD_AVAILABLE:
            ax.text(0.5, 0.5, 'WordCloud library not installed.',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Topic {topic_idx} (WordCloud unavailable)', fontweight='bold')
            return

        topic = self.topics_data[topic_idx]
        words = topic['words'][:WORDCLOUD_MAX_WORDS]
        weights = topic['weights'][:WORDCLOUD_MAX_WORDS]
        word_freq = dict(zip(words, weights))

        wordcloud = WordCloud(width=800, height=600, background_color='white',
                             colormap='viridis', max_words=WORDCLOUD_MAX_WORDS).generate_from_frequencies(word_freq)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')

        title = self.topic_labels[topic_idx]
        quality = topic.get('quality_score', 0)
        confidence = topic.get('interpretation_confidence', 'unknown')
        ax.set_title(f'{title}\n(Quality: {quality:.2f} | Confidence: {confidence})',
                    fontweight='bold', fontsize=14)