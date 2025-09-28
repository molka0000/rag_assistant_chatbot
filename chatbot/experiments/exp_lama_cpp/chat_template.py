import time
import json
import statistics
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from datetime import datetime
from typing import Dict, List, Tuple, Any
import re
import pandas as pd
import numpy as np

# Imports pour les métriques d'évaluation
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

from bot.client.lama_cpp_client import LamaCppClient
from bot.model.model_registry import get_model_settings

# Dataset Q/A simple pour l'évaluation (sans contexte - sera géré par votre prompt.py)
QA_DATASET = [
    {
        "id": 1,
        "question": "Quels documents l'assureur doit-il vous remettre avant la souscription d'un contrat ?",
        "reference_answer": "L'assureur doit remettre : une fiche d'information sur le prix et les garanties, un exemplaire du projet de contrat et ses pièces annexes ou une notice d'information détaillant les garanties, exclusions et obligations de l'assuré, et les informations sur la loi applicable et les recours possibles.",
        "article": "L. 112-2 du Code des assurances"
    },
    {
        "id": 2,
        "question": "Pouvez-vous revenir sur votre décision après avoir souscrit à distance à une assurance ?",
        "reference_answer": "Oui, le souscripteur dispose d'un délai de 14 jours pour renoncer aux contrats à distance (30 jours pour l'assurance vie). Ce droit ne s'applique pas si le contrat a été entièrement exécuté ou si un sinistre a eu lieu pendant ce délai.",
        "article": "L. 112-2-1 du Code des assurances"
    },
    {
        "id": 3,
        "question": "Que doit contenir votre police d'assurance pour être conforme ?",
        "reference_answer": "La police doit indiquer : les noms et domiciles des parties, l'objet assuré, la nature des risques, la durée et le montant de la garantie, la prime. La loi applicable si ce n'est pas la loi française et l'adresse du siège social de l'assureur. Les clauses de nullité, déchéance ou exclusion doivent être en caractères très apparents.",
        "article": "L.  L112-4 du Code des assurances"
    },
    {
        "id": 4,
        "question": "Qui supporte les pertes ou dommages causés par un cas fortuit ou par une faute de l'assuré ?",
        "reference_answer": "L'assureur prend en charge les pertes ou dommages causés par des cas fortuits ou par la faute non intentionnelle de l'assuré. Il ne couvre pas les pertes provenant d'une faute intentionnelle ou dolosive.",
        "article": "L.  L113-1 du Code des assurances"
    },
    {
        "id": 5,
        "question": "Quelles sont les obligations de l'assuré vis-à-vis de l'assureur ?",
        "reference_answer": "L'assuré doit : payer la prime aux échéances convenues, répondre exactement aux questions de l'assureur lors de la conclusion du contrat, déclarer toute aggravation ou création de risque dans les 15 jours, et signaler tout sinistre dans le délai fixé par le contrat (au moins 5 jours ouvrés, 2 jours pour vol, 24h pour mortalité du bétail).",
        "article": "L. 113-2 du Code des assurances"
    }
]

class ModelEvaluator:
    def __init__(self):
        # Initialisation des modèles pour l'évaluation
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
    def exact_match_score(self, prediction: str, reference: str) -> float:
        """Calcule le score Exact Match"""
        return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0
    
    def rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calcule les scores ROUGE"""
        scores = self.rouge_scorer.score(reference, prediction)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure
        }
    
    def bleu_score(self, prediction: str, reference: str) -> float:
        """Calcule le score BLEU"""
        try:
            # Tokenisation simple
            ref_tokens = reference.lower().split()
            pred_tokens = prediction.lower().split()
            
            smoothing = SmoothingFunction().method1
            score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
            return score
        except:
            return 0.0
    
    def meteor_score_calc(self, prediction: str, reference: str) -> float:
        """Calcule le score METEOR"""
        try:
            # Tokenisation simple pour METEOR
            ref_tokens = reference.lower().split()
            pred_tokens = prediction.lower().split()
            score = meteor_score([ref_tokens], pred_tokens)
            return score
        except:
            return 0.0
    
    def bert_score_calc(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calcule le BERTScore"""
        try:
            P, R, F1 = bert_score([prediction], [reference], lang='fr', verbose=False)
            return {
                'bert_precision': float(P[0]),
                'bert_recall': float(R[0]),
                'bert_f1': float(F1[0])
            }
        except:
            return {'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0}
    
    def semantic_similarity(self, prediction: str, reference: str) -> float:
        """Calcule la similarité sémantique avec Sentence-BERT"""
        try:
            embeddings = self.sentence_transformer.encode([prediction, reference])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def faithfulness_score(self, prediction: str, reference: str) -> float:
        """
        Évalue la fidélité de la réponse (pas d'hallucination)
        Basé sur la cohérence avec la réponse de référence
        """
        try:
            # Vérifier la cohérence avec la référence
            pred_embedding = self.sentence_transformer.encode([prediction])
            ref_embedding = self.sentence_transformer.encode([reference])
            
            similarity = cosine_similarity(pred_embedding, ref_embedding)[0][0]
            
            # Score de fidélité basé sur la similarité
            return float(max(0.0, similarity))
                
        except:
            return 0.0
    
    def answer_relevance_score(self, prediction: str, question: str) -> float:
        """
        Évalue la pertinence de la réponse par rapport à la question
        """
        try:
            pred_embedding = self.sentence_transformer.encode([prediction])
            question_embedding = self.sentence_transformer.encode([question])
            
            similarity = cosine_similarity(pred_embedding, question_embedding)[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def evaluate_response(self, prediction: str, reference: str, question: str) -> Dict[str, float]:
        """Évalue une réponse avec toutes les métriques"""
        
        results = {}
        
        # Exact Match
        results['exact_match'] = self.exact_match_score(prediction, reference)
        
        # ROUGE scores
        rouge_results = self.rouge_scores(prediction, reference)
        results.update(rouge_results)
        
        # BLEU score
        results['bleu'] = self.bleu_score(prediction, reference)
        
        # METEOR score
        results['meteor'] = self.meteor_score_calc(prediction, reference)
        
        # BERTScore
        bert_results = self.bert_score_calc(prediction, reference)
        results.update(bert_results)
        
        # Semantic similarity
        results['semantic_similarity'] = self.semantic_similarity(prediction, reference)
        
        # Faithfulness (cohérence avec la référence)
        results['faithfulness'] = self.faithfulness_score(prediction, reference)
        
        # Answer relevance
        results['answer_relevance'] = self.answer_relevance_score(prediction, question)
        
        return results

def evaluate_all_models(qa_dataset: List[Dict], evaluator: ModelEvaluator):
    """Évalue tous les modèles Groq disponibles"""
    
    # Liste des modèles Groq à tester
    model_names = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        # Ajoutez vos autres modèles Groq ici
    ]
    
    results = {}
    
    for model_name in model_names:
        print(f"\n🔍 Évaluation du modèle: {model_name}")
        print("="*50)
        
        try:
            # Initialiser le modèle Groq
            model_settings = get_model_settings(model_name)
            llm = LamaCppClient(model_folder="", model_settings=model_settings)
              
            
            model_results = []
            
            for qa_item in qa_dataset:
                print(f"Question {qa_item['id']}: {qa_item['question'][:50]}...")
                
                # Utiliser directement la question (votre prompt.py s'occupera du contexte)
                prompt = qa_item['question']
                
                # Générer la réponse
                start_time = time.time()
                try:
                    prediction = llm.generate_answer(prompt, max_new_tokens=200)
                    response_time = time.time() - start_time
                except Exception as e:
                    print(f"❌ Erreur génération: {e}")
                    prediction = "Erreur de génération"
                    response_time = 0
                
                # Évaluer la réponse (sans contexte)
                scores = evaluator.evaluate_response(
                    prediction=prediction,
                    reference=qa_item['reference_answer'],
                    question=qa_item['question']
                )
                
                # Ajouter les métadonnées
                scores.update({
                    'question_id': qa_item['id'],
                    'article': qa_item['article'],
                    'prediction': prediction,
                    'response_time': response_time,
                    'model_name': model_name
                })
                
                model_results.append(scores)
                
                # Afficher les scores principaux
                print(f"  📊 ROUGE-L: {scores['rougeL_f']:.3f} | BERT-F1: {scores['bert_f1']:.3f} | "
                      f"Semantic: {scores['semantic_similarity']:.3f} | Faithfulness: {scores['faithfulness']:.3f}")
            
            results[model_name] = model_results
            
        except Exception as e:
            print(f"❌ Erreur avec le modèle {model_name}: {e}")
            results[model_name] = []
    
    return results

def analyze_results(results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Analyse les résultats et génère un rapport"""
    
    all_data = []
    for model_name, model_results in results.items():
        for result in model_results:
            all_data.append(result)
    
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("❌ Aucune donnée à analyser")
        return df
    
    # Calculer les moyennes par modèle
    metrics = ['exact_match', 'rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu', 'meteor', 
               'bert_f1', 'semantic_similarity', 'faithfulness', 'answer_relevance', 'response_time']
    
    summary = df.groupby('model_name')[metrics].mean().round(4)
    
    print("\n" + "="*80)
    print("📈 RAPPORT DE PERFORMANCE DES MODÈLES")
    print("="*80)
    
    # Afficher le tableau complet
    print(summary.to_string())
    
    # Trouver le meilleur modèle pour chaque métrique
    print("\n🏆 MEILLEURS MODÈLES PAR MÉTRIQUE:")
    print("-"*50)
    for metric in metrics:
        if metric == 'response_time':
            best_model = summary[metric].idxmin()  # Plus rapide = mieux
            print(f"{metric:20s}: {best_model:15s} ({summary.loc[best_model, metric]:.4f}s)")
        else:
            best_model = summary[metric].idxmax()  # Plus haut = mieux
            print(f"{metric:20s}: {best_model:15s} ({summary.loc[best_model, metric]:.4f})")
    
    # Score composite
    important_metrics = ['bert_f1', 'semantic_similarity', 'faithfulness', 'answer_relevance']
    summary['composite_score'] = summary[important_metrics].mean(axis=1)
    
    print(f"\n🥇 CLASSEMENT GÉNÉRAL (score composite):")
    print("-"*50)
    ranking = summary['composite_score'].sort_values(ascending=False)
    for i, (model, score) in enumerate(ranking.items(), 1):
        print(f"{i}. {model:15s}: {score:.4f}")
    
    return df

def save_detailed_results(df: pd.DataFrame, results_dir: Path = Path("evaluation_results")):
    """Sauvegarde les résultats détaillés"""
    
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sauvegarder en Excel
    excel_path = results_dir / f"evaluation_results_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        # Résumé par modèle
        metrics = ['exact_match', 'rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu', 'meteor', 
                   'bert_f1', 'semantic_similarity', 'faithfulness', 'answer_relevance', 'response_time']
        summary = df.groupby('model_name')[metrics].mean().round(4)
        summary.to_excel(writer, sheet_name='Summary')
    
    print(f"💾 Résultats sauvegardés: {excel_path}")

if __name__ == "__main__":
    print("🚀 DÉMARRAGE DE L'ÉVALUATION DES MODÈLES GROQ")
    print("="*50)
    
    # Initialiser l'évaluateur
    evaluator = ModelEvaluator()
    
    # Lancer l'évaluation (pas besoin de model_folder pour Groq)
    results = evaluate_all_models(QA_DATASET, evaluator)
    
    # Analyser les résultats
    df = analyze_results(results)
    
    # Sauvegarder
    if not df.empty:
        save_detailed_results(df)
        
        # Afficher quelques exemples de réponses
        print(f"\n📝 EXEMPLES DE RÉPONSES:")
        print("-"*50)
        for model in df['model_name'].unique()[:2]:  # 2 premiers modèles
            print(f"\n{model.upper()}:")
            sample = df[df['model_name'] == model].iloc[0]
            print(f"Q: {QA_DATASET[0]['question']}")
            print(f"R: {sample['prediction'][:200]}...")
            print(f"Score BERT-F1: {sample['bert_f1']:.3f}")
    
    print(f"\n✅ Évaluation terminée!")