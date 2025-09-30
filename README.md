# Dataton

# Job Matching Algorithm

This project implements a job matching algorithm that connects job seekers with relevant job opportunities based on their profiles. It utilizes natural language processing techniques to analyze job descriptions and applicant profiles, generating embeddings to calculate similarity scores.

## Project Structure



## Setup Instructions




## Usage


### Example



## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Testando com Docker + FastAPI + Prometheus + Grafana

1️⃣ Subir os containers com Docker Compose  
```bash
docker-compose up -d
```

2️⃣ Verificar containers rodando  
```bash
docker ps
```

3️⃣ Testar a API FastAPI  

3.1 Testar `/` endpoint  
```bash
curl http://localhost:8000/docs
```

3.2 Testar `/predict` endpoint  
```bash
curl http://localhost:8000/predict
```

3.3 Testar `/metrics` endpoint  
```bash
curl http://localhost:8000/metrics
```

4️⃣ Testar Prometheus  

- Abra no navegador: [http://localhost:9090](http://localhost:9090)  
- Vá em **Status → Targets** e verifique se `ml_app:8000` está **UP**  
- Use **Graph** para testar queries, por exemplo: `requests_total`  

5️⃣ Testar Grafana  

- Abra no navegador: [http://localhost:3000](http://localhost:3000)  
- Login: **admin / admin**  
- Workspace: **Datathon**  
- Adicione **Data Source**: Prometheus (URL: `http://prometheus:9090`)  
- Crie **Dashboard → Panel → Query**: `requests_total`  

## Acessando o App via Streamlit

Rodar o Streamlit localmente (sem Docker)  
```bash
streamlit run app/streamlit.py
```

Acesse o Streamlit no navegador:  
[http://localhost:8501](http://localhost:8501)

 Explore a interface do app para visualizar as funcionalidades e resultados do algoritmo de matching.