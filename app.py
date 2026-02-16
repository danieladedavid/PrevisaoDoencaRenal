import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Predição de Doença Renal Crônica (DRC)",
    page_icon="🩺",
    layout="wide",
)

MODEL_DEFAULT_PATH = "Maquina_Preditiva.pkl"

FEATURES = ["sg", "htn", "hemo", "dm", "al", "appet", "rc", "pc"]

# Mapeamentos iguais ao notebook
MAP_YES_NO = {"Sim": 1, "Não": 0}
MAP_PC = {"Normal": 0, "Anormal": 1}          # pc: normal/abnormal
MAP_APPET = {"Boa": 1, "Ruim": 0}             # appet: good/poor

# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource
def load_model_from_path(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model_from_bytes(file_bytes: bytes):
    return pickle.loads(file_bytes)

def safe_predict(model, x_df: pd.DataFrame) -> int:
    """Retorna a classe prevista (0/1)."""
    pred = model.predict(x_df)[0]
    return int(pred)

def class_label(pred: int) -> str:
    # Pelo notebook: classification = 1 se "ckd", senão 0.
    return "🛑 Risco de DRC (ckd=1)" if pred == 1 else "✅ Sem indicativo de DRC (ckd=0)"

def build_input_df(
    sg: float, htn: int, hemo: float, dm: int, al: int, appet: int, rc: float, pc: int
) -> pd.DataFrame:
    data = {
        "sg": [sg],
        "htn": [htn],
        "hemo": [hemo],
        "dm": [dm],
        "al": [al],
        "appet": [appet],
        "rc": [rc],
        "pc": [pc],
    }
    return pd.DataFrame(data, columns=FEATURES)

# ----------------------------
# Sidebar: Model loading + diagnostics
# ----------------------------
st.sidebar.header("⚙️ Configurações")

with st.sidebar.expander("🔎 Diagnóstico do ambiente", expanded=False):
    st.write("Python:", sys.version.split()[0])
    try:
        import numpy as _np
        st.write("NumPy:", _np.__version__)
    except Exception as e:
        st.write("NumPy: erro ao importar:", repr(e))
    try:
        import sklearn as _sk
        st.write("scikit-learn:", _sk.__version__)
    except Exception as e:
        st.write("scikit-learn: erro ao importar:", repr(e))

st.sidebar.subheader("📦 Modelo")

model_source = st.sidebar.radio(
    "Como carregar o modelo?",
    ["Arquivo local (na pasta do projeto)", "Upload do arquivo .pkl"],
    horizontal=False
)

model = None
model_error = None

if model_source == "Arquivo local (na pasta do projeto)":
    model_path = st.sidebar.text_input("Caminho do modelo", value=MODEL_DEFAULT_PATH)
    if st.sidebar.button("Carregar modelo"):
        try:
            model = load_model_from_path(model_path)
            st.sidebar.success("Modelo carregado com sucesso!")
        except Exception as e:
            model_error = repr(e)
            st.sidebar.error("Falha ao carregar o modelo.")
            st.sidebar.code(model_error)
else:
    up = st.sidebar.file_uploader("Envie o .pkl do modelo", type=["pkl"])
    if up is not None:
        try:
            model = load_model_from_bytes(up.getvalue())
            st.sidebar.success("Modelo carregado com sucesso!")
        except Exception as e:
            model_error = repr(e)
            st.sidebar.error("Falha ao carregar o modelo.")
            st.sidebar.code(model_error)

# Tenta carregar automaticamente se for local e existir
if model is None and model_error is None and model_source == "Arquivo local (na pasta do projeto)":
    try:
        model = load_model_from_path(MODEL_DEFAULT_PATH)
    except Exception as e:
        model_error = repr(e)

# ----------------------------
# Main UI
# ----------------------------
st.title("🩺 Predição de Doença Renal Crônica (DRC)")
st.caption("Interface em Streamlit com as 8 variáveis usadas no seu notebook (sg, htn, hemo, dm, al, appet, rc, pc).")

if model is None:
    st.warning("Modelo não carregado. Use a barra lateral para carregar o arquivo do modelo.")
    if model_error:
        st.error("Erro ao carregar o modelo (provável incompatibilidade de versões do NumPy/sklearn com o pickle).")
        st.code(model_error)
        st.info(
            "Se o erro for de versão: a forma mais estável é re-treinar e salvar novamente no mesmo ambiente "
            "onde o app roda, ou padronizar as versões usadas no treino e no app."
        )
    st.stop()

tab1, tab2, tab3 = st.tabs(["🧍 Predição individual", "📄 Predição em lote (CSV)", "ℹ️ Sobre / Ajuda"])

# ----------------------------
# Tab 1: single prediction
# ----------------------------
with tab1:
    colA, colB = st.columns([1.2, 1])

    with colA:
        st.subheader("📋 Preencha os dados")
        with st.form("form_single"):
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                sg = st.selectbox(
                    "sg (specific gravity)",
                    options=[1.005, 1.010, 1.015, 1.020, 1.025],
                    index=3
                )
                al = st.selectbox("al (albumin)", options=[0, 1, 2, 3, 4, 5], index=0)

            with c2:
                htn_txt = st.selectbox("htn (hipertensão)", options=["Não", "Sim"], index=0)
                dm_txt = st.selectbox("dm (diabetes)", options=["Não", "Sim"], index=0)

            with c3:
                pc_txt = st.selectbox("pc (pus cell)", options=["Normal", "Anormal"], index=0)
                appet_txt = st.selectbox("appet (apetite)", options=["Boa", "Ruim"], index=0)

            with c4:
                hemo = st.number_input("hemo (hemoglobina)", min_value=0.0, max_value=25.0, value=13.0, step=0.1)
                rc = st.number_input("rc (red blood cell count)", min_value=0.0, max_value=10.0, value=4.5, step=0.1)

            submitted = st.form_submit_button("🔍 Prever")

    with colB:
        st.subheader("✅ Resultado")
        st.write("Clique em **Prever** para ver a saída do modelo.")
        st.markdown("---")

    if submitted:
        # Aplicar mapeamentos
        htn = MAP_YES_NO[htn_txt]
        dm = MAP_YES_NO[dm_txt]
        pc = MAP_PC[pc_txt]
        appet = MAP_APPET[appet_txt]

        x = build_input_df(sg=sg, htn=htn, hemo=hemo, dm=dm, al=al, appet=appet, rc=rc, pc=pc)

        st.markdown("### Entrada usada")
        st.dataframe(x, use_container_width=True)

        try:
            pred = safe_predict(model, x)

            st.markdown("### Predição")
            st.success(class_label(pred))

        except Exception as e:
            st.error("Erro ao prever. Verifique se o modelo espera exatamente essas colunas/tipos.")
            st.code(repr(e))

# ----------------------------
# Tab 2: batch prediction
# ----------------------------
with tab2:
    st.subheader("📄 Predição em lote (CSV)")
    st.write("Envie um CSV com **as 8 colunas**: `sg, htn, hemo, dm, al, appet, rc, pc`.")

    st.markdown("**Formato esperado:**")
    st.code(",".join(FEATURES))

    up_csv = st.file_uploader("Upload do CSV", type=["csv"], key="csv_uploader")

    if up_csv is not None:
        try:
            df = pd.read_csv(up_csv)

            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                st.error(f"Seu CSV não tem as colunas: {missing}")
                st.stop()

            # Garante ordem e tipos numéricos quando possível
            Xb = df[FEATURES].copy()

            # Se vierem textos (Sim/Não etc.), tenta mapear automaticamente
            # (não é perfeito, mas ajuda)
            def map_auto(s: pd.Series):
                if s.dtype == object:
                    s2 = s.astype(str).str.strip().str.lower()
                    # yes/no
                    s2 = s2.replace({"sim": 1, "não": 0, "nao": 0, "yes": 1, "no": 0})
                    # appet
                    s2 = s2.replace({"boa": 1, "ruim": 0, "good": 1, "poor": 0})
                    # pc
                    s2 = s2.replace({"normal": 0, "anormal": 1, "abnormal": 1})
                    return pd.to_numeric(s2, errors="coerce")
                return pd.to_numeric(s, errors="coerce")

            for c in FEATURES:
                Xb[c] = map_auto(Xb[c])

            # Aviso de NaNs
            if Xb.isna().any().any():
                st.warning("Alguns valores viraram NaN após conversão. Verifique seu CSV (tipos/categorias).")

            preds = model.predict(Xb)

            out = df.copy()
            out["pred"] = preds.astype(int)
            out["pred_label"] = out["pred"].apply(class_label)

            st.success("Predições geradas!")
            st.dataframe(out, use_container_width=True)

            # download
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Baixar resultados (.csv)",
                data=csv_bytes,
                file_name="predicoes_doenca_renal.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("Falha no processamento do CSV.")
            st.code(repr(e))

# ----------------------------
# Tab 3: about/help
# ----------------------------
with tab3:
    st.subheader("ℹ️ Como este app funciona")
    st.write(
        "Este app usa um modelo de Machine Learning treinado para classificar o risco de "
        "Doença Renal Crônica (DRC) a partir de 8 variáveis clínicas. Preencha os campos "
        "na aba de predição individual ou envie um CSV na aba de lote para receber as "
        "predições. A saída `ckd=1` indica maior risco, enquanto `ckd=0` indica ausência "
        "de indicativo de DRC."
    )
