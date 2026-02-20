# Predição de Doença Renal Crônica (DRC)

Aplicação de **Machine Learning** para predição de **Doença Renal Crônica (DRC)**, com interface em **Streamlit** 

## 🧩 Estrutura sugerida do repositório

```
├─ app.py                      # App Streamlit
├─ PrevDoencaRenais_notebook.ipynb
├─ Kidney_data.csv             # Dataset
├─ Maquina_Preditiva.pkl       # Modelo 
├─ requirements.txt
├─ .gitignore
└─ LICENSE
```


## 🚀 Como rodar (Streamlit)

### 1) Criar e ativar ambiente virtual

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
python3 -m venv.venv
source .venv/bin/activate
```

### 2) Instalar dependências
```bash
pip install -r requirements.txt
```

### 3) Rodar o app
```bash
streamlit run app.py
```


## 📝 Licença
Este projeto está sob a licença **MIT** — veja o arquivo `LICENSE`.

## Autoria
CDPRO - Daniela de David