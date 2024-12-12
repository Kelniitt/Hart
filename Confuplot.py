import seaborn as sns

from sklearn.metrics import confusion_matrix

def Confuplot(label, logits, labeler, **plotargs):
  """
  Confusion Matrix Plot

  >>> logits = model.predict(...)
  >>> Confuplot(label=label, logits=logits, labeler=labeler, cmap="Pastel1", fmt="g")
  """
  cm = confusion_matrix(label, logits)
  ploti = sns.heatmap(cm, annot=True, **plotargs);
  ploti.set_xlabel("Prediction")
  ploti.set_ylabel("True")
  ploti.set_xticklabels(labeler)
  ploti.set_yticklabels(labeler)
  ploti.set_title("Confusion Matrix");
