import streamlit as st
import pandas as pd
import os
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import json
from datetime import datetime

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(page_title="Assistant Code Python RAG",
                   page_icon="🐍",
                   layout="wide")

st.title("🐍 Assistant Code Python RAG (Basé sur BDD CSV)")
st.markdown("Ce RAG utilise votre CSV question,code pour répondre aux questions de programmation Python, indexé par Chroma et généré par Ollama.")

# --- Constantes pour la nouvelle BDD ---
QUESTION_COL = "question"
CODE_COL = "code"
DEFAULT_CSV_PATH = "Python_codes.csv"
HISTORY_FILE = "conversation_history.json"

# =========================
# HELPERS / UTILS
# =========================
def try_read_csv(path):
    """ Tente de lire le CSV avec différents séparateurs. """
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=';')
        except:
            return None

@st.cache_data
def load_dataset(local_path: str = None):
    """ Charge le dataset à partir du chemin local. """
    if local_path and os.path.exists(local_path):
        df = try_read_csv(local_path)
        return df
    return None

def build_soup(row: pd.Series, question_col: str, code_col: str) -> str:
    """
    Construit la "soup" de texte à partir des colonnes Question et Code.
    """
    question = str(row.get(question_col, "")).strip()
    code = str(row.get(code_col, "")).strip()
    return f"Question Python: {question}\nCode Python associé: {code}"

# =========================
# GESTION DE L'HISTORIQUE
# =========================
def load_conversation_history():
    """Charge l'historique des conversations depuis le fichier JSON."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_conversation_history(history):
    """Sauvegarde l'historique des conversations dans un fichier JSON."""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde de l'historique: {e}")

def add_to_history(user_msg, assistant_msg, sources):
    """Ajoute une interaction à l'historique persistant."""
    if 'persistent_history' not in st.session_state:
        st.session_state.persistent_history = load_conversation_history()
    
    entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'user': user_msg,
        'assistant': assistant_msg,
        'sources': sources
    }
    
    st.session_state.persistent_history.append(entry)
    save_conversation_history(st.session_state.persistent_history)

def clear_history():
    """Efface l'historique des conversations."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    st.session_state.persistent_history = []
    st.session_state.messages = []

# =========================
# CHARGEMENT DES DONNÉES (UI)
# =========================
st.sidebar.header("Chargement du dataset")
st.sidebar.markdown(f"Charge le CSV contenant les colonnes **'{QUESTION_COL}'** et **'{CODE_COL}'**.")

local_path_input = st.sidebar.text_input("Chemin local du CSV (optionnel)", value=DEFAULT_CSV_PATH)
use_upload = st.sidebar.checkbox("Je veux uploader un fichier CSV maintenant", value=False)

df = None
if use_upload:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success("CSV uploadé avec succès.")
        except Exception as e:
            st.error(f"Impossible de lire l'upload : {e}")
else:
    df = load_dataset(local_path_input)

if df is None:
    st.warning("Aucun dataset chargé. Charge un CSV local ou coche 'Je veux uploader' pour fournir un fichier.")
    st.stop()

if QUESTION_COL not in df.columns or CODE_COL not in df.columns:
    st.error(f"Le fichier CSV doit contenir les colonnes '{QUESTION_COL}' et '{CODE_COL}'. Colonnes trouvées : {list(df.columns)}")
    st.stop()
    
st.write(f"Dataset chargé ({len(df)} lignes). Aperçu des colonnes clés :")
st.dataframe(df[[QUESTION_COL, CODE_COL]].head(3))

with st.spinner("Construction des champs pour l'indexation..."):
    df['soup'] = df.apply(lambda r: build_soup(r, QUESTION_COL, CODE_COL), axis=1)
    
    if 'id' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'id'})
    
    st.success("Soup créée. Exemple de contenu indexé :")

example_cols = ['id', QUESTION_COL, 'soup']
st.dataframe(df[example_cols].head(5))

# =========================
# INITIALISER RAG
# =========================
@st.cache_resource
def init_rag(df: pd.DataFrame, collection_path: str = "./rag_chroma_db_python", collection_name: str = "python_codes_rag") -> tuple[chromadb.Collection, SentenceTransformer]:
    
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    chroma_client = chromadb.PersistentClient(path=collection_path)

    try:
        collection = chroma_client.get_collection(name=collection_name)
        if collection.count() == 0:
            raise Exception("Collection existante mais vide. Re-création.")
    except Exception:
        collection = chroma_client.create_collection(name=collection_name)
        
        st.info(f"Création de la collection Chroma à partir de {len(df)} documents. Cela peut prendre un moment.")
        
        texts = df['soup'].astype(str).tolist()
        embeddings = model.encode(texts, show_progress_bar=True)

        metadatas = []
        ids = []
        for idx, row in df.iterrows():
            ids.append(str(idx)) 
            meta = {
                'question_originale': str(row.get(QUESTION_COL, "")),
                'code_snippet': str(row.get(CODE_COL, ""))
            }
            metadatas.append(meta)

        BATCH_SIZE = 5000
        total_docs = len(ids)
        
        progress_bar = st.progress(0)
        for i in range(0, total_docs, BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, total_docs)
            
            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx].tolist()
            batch_metadatas = metadatas[i:end_idx]
            batch_documents = texts[i:end_idx]
            
            collection.add(
                ids=batch_ids, 
                embeddings=batch_embeddings, 
                metadatas=batch_metadatas, 
                documents=batch_documents
            )
            
            progress = (end_idx / total_docs)
            progress_bar.progress(progress)
            st.write(f"Indexé {end_idx}/{total_docs} documents...")
        
        progress_bar.empty()

    return collection, model

with st.spinner("Initialisation du pipeline RAG (Chroma + embeddings)..."):
    collection, embed_model = init_rag(df)
    st.success(f"Index RAG prêt. {collection.count()} documents indexés.")

# =========================
# FONCTION OLLAMA
# =========================
def query_ollama(context: str, question: str, conversation_history: List[Dict], model_name: str = "qwen2.5:1.5b") -> str:
    """
    Envoie prompt à Ollama avec contexte RAG et historique de conversation.
    """
    url = "http://localhost:11434/api/generate"
    
    # Construire l'historique récent (derniers 3 échanges)
    history_context = ""
    if conversation_history:
        recent_history = conversation_history[-3:]  # Limiter à 3 derniers échanges
        for msg in recent_history:
            history_context += f"User: {msg['content']}\n"
            if msg['role'] == 'assistant':
                history_context += f"Assistant: {msg['content']}\n"
    
    prompt = f"""[ROLE] Expert en Programmation Python.

[HISTORIQUE DE CONVERSATION]
{history_context if history_context else "Pas d'historique précédent."}

[CONTEXTE RAG]
Voici des exemples de code Python pertinents trouvés dans ma base de connaissances:

{context}

[QUESTION ACTUELLE]
{question}

[INSTRUCTION]
Répondez à la question en tenant compte de l'historique de la conversation si pertinent. Si le contexte contient du code pertinent, incluez-le dans un bloc de code Python (```python...```) et expliquez-le brièvement. Fournissez la réponse finale en français.
"""
    try:
        response = requests.post(url, json={"model": model_name, "prompt": prompt, "stream": False, "temperature": 0.2})
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and 'response' in data:
                return data['response']
            return str(data)
        else:
            return f"Erreur Ollama ({response.status_code}): {response.text}"
    except Exception as e:
        return f"Erreur de connexion à Ollama: {e}. Assurez-vous qu'Ollama est en cours d'exécution."

# =========================
# INTERFACE DE CHAT
# =========================
model_choice = "qwen2.5:1.5b"

# Gestion de l'historique
st.sidebar.header("Historique")
if st.sidebar.button("🗑️ Effacer l'historique"):
    clear_history()
    st.rerun()

if st.sidebar.button("📥 Télécharger l'historique"):
    if 'persistent_history' in st.session_state and st.session_state.persistent_history:
        history_json = json.dumps(st.session_state.persistent_history, ensure_ascii=False, indent=2)
        st.sidebar.download_button(
            label="💾 Sauvegarder JSON",
            data=history_json,
            file_name=f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.sidebar.warning("Aucun historique à télécharger")

# Affichage du nombre de conversations
if 'persistent_history' not in st.session_state:
    st.session_state.persistent_history = load_conversation_history()

st.sidebar.metric("Conversations sauvegardées", len(st.session_state.persistent_history))

# Initialisation des messages de session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique persistant dans une expander
with st.sidebar.expander("📜 Voir l'historique complet"):
    if st.session_state.persistent_history:
        for idx, entry in enumerate(reversed(st.session_state.persistent_history[-10:])):  # Dernières 10
            st.markdown(f"**{entry['timestamp']}**")
            st.markdown(f"👤 User: {entry['user'][:50]}...")
            st.markdown(f"🤖 Assistant: {entry['assistant'][:50]}...")
            st.divider()
    else:
        st.info("Aucun historique disponible")

# Afficher messages de la session actuelle
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Pose ta question Python (ex: 'Comment puis-je sommer toutes les valeurs dans un dictionnaire ?')")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status = st.status("Recherche RAG en cours...", expanded=True)

        # Retrieval
        q_emb = embed_model.encode([user_input])
        try:
            results = collection.query(query_embeddings=q_emb.tolist(), n_results=5)
        except Exception as e:
            st.error(f"Erreur lors de la recherche dans Chroma: {e}")
            results = None

        context_txt = ""
        sources_found = []
        
        if results and 'ids' in results and len(results['ids'])>0:
            metas = results.get('metadatas', [[]])[0]
            docs = results.get('documents', [[]])[0]
            
            for i, meta in enumerate(metas):
                full_document = docs[i]
                context_txt += f"--- Document {i+1} ---\n{full_document}\n\n"
                
                q_snip = meta.get('question_originale', 'Question non disponible')
                c_snip = meta.get('code_snippet', 'Code non disponible')
                status.write(f"Source trouvée : **{q_snip[:50]}...**")
                
                sources_found.append({
                    'question': q_snip,
                    'code_preview': c_snip[:100]
                })
        else:
            status.write("Aucun résultat pertinent trouvé dans l'index. Le LLM répondra avec ses connaissances générales.")

        status.update(label="RAG terminé", state="complete", expanded=False)

        # Génération via Ollama avec historique
        resp_placeholder = st.empty()
        resp_placeholder.markdown("🤖 *Le LLM Expert Python réfléchit...*")
        llm_answer = query_ollama(context_txt, user_input, st.session_state.messages, model_choice)
        resp_placeholder.markdown(llm_answer)

        st.session_state.messages.append({"role": "assistant", "content": llm_answer})
        
        # Sauvegarder dans l'historique persistant
        add_to_history(user_input, llm_answer, sources_found)