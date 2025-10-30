import gradio as gr
from transformers import pipeline

APP_NAME = "MoodMapper"
MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"  # si mémoire faible, essaie: "SamLowe/roberta-base-go_emotions"

_clf = None
def get_clf():
    global _clf
    if _clf is None:
        _clf = pipeline("text-classification", model=MODEL_ID, return_all_scores=True, truncation=True)
    return _clf

def predict_emotion(text):
    if not text or not text.strip():
        return {"": 0.0}, "Veuillez entrer un texte."
    scores = get_clf()(text)[0]
    score_dict = {s["label"]: float(s["score"]) for s in scores}
    top = max(scores, key=lambda d: d["score"])
    notes = f"**Émotion prédite :** {top['label']} ({top['score']*100:.1f}%)"
    return score_dict, notes

with gr.Blocks(title=APP_NAME) as demo:
    gr.Markdown(f"# {APP_NAME} — Détecteur d'émotions (texte)")
    txt = gr.Textbox(label="Votre texte", placeholder="Ex: Je suis déçue et un peu en colère par ce mail.", lines=3)
    btn = gr.Button("Analyser")
    out_scores = gr.Label(label="Scores (confiances)")
    out_notes = gr.Markdown()
    gr.Examples(inputs=txt, examples=[
        "I just got the internship — I'm so happy!",
        "Je suis déçue et un peu en colère par ce mail.",
        "This is fine, nothing special today.",
        "I'm worried about the deadline...",
        "Quelle surprise ! Je ne m’y attendais pas."
    ], label="Exemples")
    btn.click(predict_emotion, txt, [out_scores, out_notes])
    txt.submit(predict_emotion, txt, [out_scores, out_notes])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
