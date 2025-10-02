# 📊 **Backup e Restore dos Dashboards Grafana**

## 🚨 **IMPORTANTE: SEMPRE FAÇA BACKUP!**

### **📤 Como Exportar Dashboard:**

1. **Abra o Grafana:** http://localhost:3000
2. **Acesse seu dashboard**
3. **Clique no ícone de configuração** (⚙️) no topo direito
4. **"JSON Model"**
5. **Copie todo o JSON**
6. **Salve em:** `grafana/dashboards/[nome-do-dashboard].json`

### **📥 Como Importar Dashboard:**

#### **Método 1: Via Interface Grafana**
1. **"+" (Plus) → "Import"**
2. **Cole o JSON ou faça upload do arquivo**
3. **Configure o nome e pasta**
4. **"Import"**

#### **Método 2: Via Provisioning (Recomendado)**
1. **Salve o JSON em:** `grafana/dashboards/`
2. **Reinicie o container:**
   ```powershell
   docker restart grafana
   ```

### **📂 Estrutura de Arquivos:**

```
grafana/
├── provisioning/
│   ├── datasources/
│   │   └── prometheus.yml        # ✅ Configuração do Prometheus
│   └── dashboards/
│       └── dashboard.yml         # ✅ Configuração de carregamento
└── dashboards/
    ├── job-matching-dashboard.json           # ❌ Arquivo antigo (simples)
    └── sistema-job-matching-dashboard.json   # ✅ Arquivo novo (completo)
```

### **🔄 Workflow Recomendado:**

1. **Desenvolver dashboard no Grafana**
2. **Exportar como JSON**
3. **Salvar no repositório**
4. **Commit no Git**
5. **Reiniciar container para aplicar**

### **⚠️ Cuidados:**

- **Sempre exporte** após fazer mudanças
- **Use nomes descritivos** para os arquivos
- **Mantenha versionamento** no Git
- **Teste a importação** em ambiente limpo

### **🔧 Comandos Úteis:**

```powershell
# Reiniciar apenas o Grafana
docker restart grafana

# Ver logs do Grafana
docker logs grafana

# Backup completo do volume
docker run --rm -v dataton_grafana-data:/source -v ${PWD}:/backup alpine tar czf /backup/grafana-backup.tar.gz -C /source .
```

### **📝 Template para Novos Dashboards:**

Sempre inclua no JSON:
- **UID único:** `"uid": "dashboard-name-uid"`
- **Título descritivo:** `"title": "Sistema XYZ - Monitoramento"`
- **Tags organizacionais:** `"tags": ["sistema", "monitoring"]`
- **Datasource correto:** `"datasource": "Prometheus"`