# Churn MLOps

Projeto simples de predição de churn com FastAPI e um modelo treinado em scikit-learn.

Estrutura principal

- `api/` - aplicação FastAPI que serve o formulário estático e expõe a rota `/predict`.
- `static/` - arquivos estáticos servidos (ex.: `form.html`).
- `model/` - artefato `model.pkl` (contém modelo + scaler + encoders).
- `form.html` - formulário frontend (estilizado) para enviar dados ao endpoint `/predict`.
- `Dockerfile` - imagem para executar a API com `uvicorn`.
- `requirements.txt` - dependências Python.

Requisitos

- Docker (opcional, usado para rodar a API em container)
- Python 3.10+ (se for executar localmente sem Docker)

Instalação e execução local (sem Docker)

1. Crie um ambiente virtual (recomendado):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instale dependências:

```bash
pip install -r requirements.txt
```

3. Execute a API com uvicorn (a partir da raiz do projeto):

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

4. Abra o navegador em `http://127.0.0.1:8000/` para ver o formulário estilizado.

Executando via Docker

1. Build da imagem (a partir da raiz do projeto):

```bash
docker build -t churn-api .
```

2. Rodar o container:

```bash
docker run -p 8000:8000 churn-api
```

3. Abra `http://localhost:8000/` no navegador.

Endpoint `/predict`

- Método: POST
- URL: `http://localhost:8000/predict`
- Body (JSON):

```json
{
  "credit_score": 650,
  "country": "France",
  "gender": "Female",
  "age": 35,
  "tenure": 5,
  "balance": 12000.5,
  "products_number": 2,
  "credit_card": 1,
  "active_member": 1,
  "estimated_salary": 50000
}
```

- Resposta (exemplo):

```json
{
  "churn_predito": 0,
  "probabilidade_de_churn": 0.1234
}
```

Testando o frontend

- Abra `http://localhost:8000/` e preencha o formulário.
- Clique em "Enviar" — o frontend faz um POST para `/predict` e exibirá o resultado formatado com probabilidade e cor de risco.

Observações e troubleshooting

- Se receber mensagens de erro relacionadas ao modelo (por exemplo, país ou gênero desconhecido), verifique se os valores usados no formulário correspondem aos usados no treino (os encoders aceitam apenas categorias vistas durante o treino).
- Se rodando via Docker e a porta 8000 já estiver ocupada, altere o mapeamento do container (por exemplo `-p 8001:8000`) e acesse `http://localhost:8001/`.
- O arquivo `model/model.pkl` é esperado estar presente. Se faltar, a API não iniciará corretamente.

Contribuições

Pull requests são bem-vindos. Para mudanças no frontend, edite `static/form.html` (o servidor serve este arquivo).

Licença

Projeto para fins educacionais.
