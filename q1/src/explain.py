import shap

def shap_summary(trained_pipeline, X_sample):
    # Works best with tree-based models; ensure final step is classifier named 'clf'
    model = trained_pipeline.named_steps['clf']
    pre = trained_pipeline.named_steps['pre']
    X_enc = pre.transform(X_sample)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_enc)
        shap.summary_plot(shap_values, X_enc, show=False)
    except Exception:
        # Generic fallback (can be slow); consider sampling
        explainer = shap.KernelExplainer(model.predict_proba, X_enc[:100] if hasattr(X_enc, 'toarray') else X_enc[:100])
        shap_values = explainer.shap_values(X_enc[:200])
        shap.summary_plot(shap_values[1], X_enc[:200], show=False)
