import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 1. CONFIGURAÇÃO DA PÁGINA (Substitui o geometry e bg do Tkinter)
# O modo "wide" expande o painel para ocupar toda a largura da tela do navegador
st.set_page_config(
    page_title="Análise de Qualidade de Serviços - IA", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Cabeçalho Superior Personalizado
st.markdown(
    """
    <div style="background-color:#FF8C00; padding:18px; border-radius:8px; margin-bottom:25px;"> #2c3e50
        <h2 style="color:white; text-align:center; margin:0; font-family:Helvetica;">
            Painel de Diagnóstico de Serviços & Treinamento (IA)
        </h2>
    </div>
    """,
    unsafe_allow_html=True
)

# 2. MOTOR DE INTELIGÊNCIA ARTIFICIAL (Mantido estritamente igual)
def simular_dados_treinamento():
    """Gera a inteligência base da IA (Padrões de Negócio)"""
    np.random.seed(42)
    X_treino, y_treino = [], []

    for _ in range(250):
        reclamacoes = np.random.randint(0, 50)
        tempo = np.random.uniform(0.5, 5.0)
        satisfacao = np.random.uniform(1.0, 5.0)

        # Critérios de classificação em níveis para a rede memorizar
        if reclamacoes > 35 or satisfacao < 2.0:
            nivel = 0  # Serviço Muito Ruim
        elif reclamacoes > 20 or satisfacao < 3.5:
            nivel = 1  # Serviço Ruim
        elif reclamacoes > 7 or satisfacao < 4.4:
            nivel = 2  # Serviço Bom
        else:
            nivel = 3  # Serviço Excelente

        X_treino.append([reclamacoes, tempo, satisfacao])
        y_treino.append(nivel)

    return np.array(X_treino), np.array(y_treino)


# 3. INTERFACE DO USUÁRIO (Área de Upload)
st.subheader("1. Base de Dados")
arquivo_carregado = st.file_uploader("Selecione o arquivo de Serviços (Formato CSV)", type=["csv"])

# Lógica de controle de estado (Substitui as travas de botões do Tkinter)
if arquivo_carregado is not None:
    st.success(f"Arquivo '{arquivo_carregado.name}' importado com sucesso!")
    
    # Botão de Ação (Aparece em destaque após o upload do arquivo)
    if st.button("📊 Executar Análise Preditiva (Rede Neural)", type="primary"):
        
        # O 'st.spinner' mostra uma animação profissional de carregamento
        with st.spinner("A Rede Neural está processando os dados operacionais..."):
            try:
                # 1. Carrega o arquivo dinâmico enviado pelo usuário
                dados_reais = pd.read_csv(arquivo_carregado)

                # Validação técnica das colunas do arquivo enviado
                colunas_obrigatorias = {
                    "servico",
                    "reclamacoes",
                    "tempo_resolucao_horas",
                    "satisfacao_cliente",
                }
                
                if not colunas_obrigatorias.issubset(dados_reais.columns):
                    st.error(
                        "Erro na estrutura do arquivo! O CSV precisa conter exatamente as colunas:\n"
                        "servico, reclamacoes, tempo_resolucao_horas, satisfacao_cliente"
                    )
                else:
                    # 2. Treinamento em tempo de execução da Rede Neural
                    X_treino, y_treino = simular_dados_treinamento()

                    scaler = StandardScaler()
                    X_treino_scaled = scaler.fit_transform(X_treino)

                    # Configurando o Classificador Perceptron Multicamadas (MLP)
                    mlp = MLPClassifier(
                        hidden_layer_sizes=(12, 12),
                        max_iter=1200,
                        random_state=42,
                        learning_rate_init=0.01,
                    )
                    mlp.fit(X_treino_scaled, y_treino)

                    # 3. Predição com a base carregada
                    X_reais = dados_reais[
                        ["reclamacoes", "tempo_resolucao_horas", "satisfacao_cliente"]
                    ].values
                    X_reais_scaled = scaler.transform(X_reais)

                    predicoes = mlp.predict(X_reais_scaled)

                    # Mapeamento do formato de resposta solicitado
                    mapeamento_status = {
                        0: "Serviço Muito Ruim",
                        1: "Serviço Ruim",
                        2: "Serviço Bom",
                        3: "Serviço Excelente",
                    }

                    mapeamento_acao = {
                        0: "⚠️ Alerta Vermelho: Necessidade crítica de treinamento e auditoria.",
                        1: "🛑 Treinamento Necessário: Reciclagem operacional recomendada.",
                        2: "✅ Sob Controle: Manter acompanhamento preventivo regular.",
                        3: "⭐ Excelente: Padrão ouro. Utilize como modelo de benchmark.",
                    }

                    # 4. Construção dos resultados para exibição tabular
                    linhas_resultados = []
                    for i, linha in dados_reais.iterrows():
                        classe_idx = predicoes[i]
                        linhas_resultados.append({
                            "Serviço Analisado": linha["servico"],
                            "Nível de Qualidade": mapeamento_status[classe_idx],
                            "Ação Recomendada": mapeamento_acao[classe_idx]
                        })
                    
                    df_final = pd.DataFrame(linhas_resultados)

                    # Área de Resultados (Substitui o Treeview/Tabela do Tkinter)
                    st.write("---")
                    st.subheader("2. Classificação Gerencial de Desempenho")
                    
                    # Exibe os dados de forma elegante e interativa na tela
                    st.dataframe(
                        df_final, 
                        use_container_width=True, 
                        hide_index=True
                    )

                    # Notificação flutuante de sucesso
                    st.toast("Mapeamento operacional concluído!")
                    st.success("A inteligência artificial concluiu a análise estratégica com sucesso!")

            except Exception as e:
                st.error(f"Ocorreu um erro no processamento dos dados. Detalhes: {e}")
else:
    st.info("Aguardando a inserção do arquivo de dados para iniciar a auditoria de inteligência.")
    
