# app.py - Dashboard Internações DATASUS
# CORRIGIDO: use_container_width → width='stretch' e px.barh → px.bar com orientation

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ============================================
# 🎨 CONFIGURAÇÃO DA PÁGINA
# ============================================

st.set_page_config(
    page_title="Dashboard Internações DATASUS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 🎨 CORES (PALETA COORDENADA)
# ============================================

DARK_BLUE = '#1E2761'      # Navy
TEAL = '#02B090'           # Teal
LIGHT_BLUE = '#CADCFC'    # Ice Blue
WHITE = '#FFFFFF'          # White
GRAY = '#505050'           # Gray
CORAL = '#F96167'          # Coral
GOLD = '#F9E795'           # Gold

# ============================================
# 📊 CARREGAR DADOS (COM CACHE)
# ============================================

@st.cache_data
def load_data():
    """
    Carrega dados de internações
    """
    
    np.random.seed(42)
    n_records = 5000
    
    # Datas
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days
    random_dates = [start_date + timedelta(days=np.random.randint(0, date_range)) for _ in range(n_records)]
    
    # Categorical
    ufs = ['SP', 'RJ', 'MG', 'BA', 'RS', 'PE', 'PR', 'SC', 'GO', 'DF']
    especialidades = {
        '01': 'Clínica Médica',
        '02': 'Cirurgia Geral',
        '03': 'Ginecologia',
        '04': 'Obstetrícia',
        '05': 'Cardiologia',
        '06': 'Neurologia',
        '07': 'Pediatria',
        '08': 'Traumatologia',
        '09': 'Ortopedia',
        '10': 'Oncologia'
    }
    
    # Create DataFrame
    df = pd.DataFrame({
        'N_AIH': [f'{i:010d}' for i in range(1, n_records + 1)],
        'ANO_CMPT': [2023] * n_records,
        'MES_CMPT': [d.month for d in random_dates],
        'DT_INTER': random_dates,
        'UF_ZI': np.random.choice(ufs, n_records),
        'ESPEC': np.random.choice(list(especialidades.keys()), n_records),
        'SEXO': np.random.choice([1, 3], n_records),
        'IDADE': np.random.gamma(shape=2, scale=20, size=n_records).astype(int),
        'DIAS_PERM': np.random.exponential(scale=5, size=n_records).astype(int) + 1,
        'CAR_INT': np.random.choice([1, 4], n_records, p=[0.4, 0.6]),
        'MORTE': np.random.binomial(1, 0.02, n_records),
        'ESPEC_NOME': [especialidades.get(str(x), 'Outra') for x in np.random.choice(list(especialidades.keys()), n_records)],
        'SEXO_NOME': np.random.choice(['Masculino', 'Feminino'], n_records),
        'CAR_INT_NOME': np.random.choice(['Eletiva', 'Urgência'], n_records),
        'VAL_SH': np.random.gamma(2, 500, n_records),
        'VAL_SP': np.random.gamma(2, 300, n_records),
        'VAL_SADT': np.random.gamma(2, 200, n_records),
        'VAL_UTI': np.random.exponential(1000, n_records),
        'NUM_PROC': np.random.poisson(3, n_records),
        'MARCA_UTI': np.random.binomial(1, 0.15, n_records),
    })
    
    df['VAL_TOT'] = df['VAL_SH'] + df['VAL_SP'] + df['VAL_SADT'] + df['VAL_UTI']
    df['DT_SAIDA'] = df.apply(lambda row: row['DT_INTER'] + timedelta(days=row['DIAS_PERM']), axis=1)
    
    return df

df = load_data()

# ============================================
# 📱 TÍTULO E LAYOUT
# ============================================

st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #1E2761;
    font-size: 2.5em;
    margin-bottom: 0.5em;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #505050;
    font-size: 1.2em;
    margin-bottom: 1em;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🏥 Dashboard de Internações - DATASUS</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Análise Completa de Dados em Tempo Real</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# 🔍 FILTROS (SIDEBAR)
# ============================================

st.sidebar.title("🔍 Filtros")

uf_selecionada = st.sidebar.multiselect(
    "Selecione UF:",
    sorted(df['UF_ZI'].unique()),
    default=sorted(df['UF_ZI'].unique())[:3]
)

especialidade_selecionada = st.sidebar.multiselect(
    "Selecione Especialidade:",
    sorted(df['ESPEC_NOME'].unique()),
    default=sorted(df['ESPEC_NOME'].unique())[:3]
)

carater_selecionado = st.sidebar.multiselect(
    "Selecione Caráter:",
    sorted(df['CAR_INT_NOME'].unique()),
    default=sorted(df['CAR_INT_NOME'].unique())
)

df_filtrado = df[
    (df['UF_ZI'].isin(uf_selecionada)) &
    (df['ESPEC_NOME'].isin(especialidade_selecionada)) &
    (df['CAR_INT_NOME'].isin(carater_selecionado))
]

st.sidebar.markdown("---")
st.sidebar.info(f"📊 Registros mostrados: {len(df_filtrado):,} de {len(df):,}")

# ============================================
# 📊 KPIs PRINCIPAIS
# ============================================

st.markdown("### 🎯 Indicadores Principais")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_internacoes = len(df_filtrado)
    st.metric("📋 Total de Internações", f"{total_internacoes:,}")

with col2:
    taxa_mortalidade = (df_filtrado['MORTE'].sum() / len(df_filtrado) * 100) if len(df_filtrado) > 0 else 0
    st.metric("💀 Taxa de Mortalidade", f"{taxa_mortalidade:.2f}%")

with col3:
    dias_medio = df_filtrado['DIAS_PERM'].mean() if len(df_filtrado) > 0 else 0
    st.metric("📅 Dias Permanência (Média)", f"{dias_medio:.1f}")

with col4:
    custo_medio = df_filtrado['VAL_TOT'].mean() if len(df_filtrado) > 0 else 0
    st.metric("💰 Custo Médio", f"R$ {custo_medio:,.0f}")

st.markdown("---")

# ============================================
# 📈 GRÁFICOS (LINHA 1)
# ============================================

col1, col2 = st.columns(2)

# Gráfico 1: Internações por Mês
with col1:
    mes_data = df_filtrado.groupby('MES_CMPT').size().reset_index(name='Total')
    meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    mes_data['Mês'] = mes_data['MES_CMPT'].map(lambda x: meses[x-1] if x <= 12 else 'N/A')
    
    fig1 = px.bar(
        mes_data,
        x='Mês',
        y='Total',
        title='📊 Internações por Mês',
        color_discrete_sequence=[TEAL],
        text='Total'
    )
    fig1.update_traces(textposition='auto')
    fig1.update_layout(
        height=400,
        hovermode='x unified',
        paper_bgcolor=WHITE,
        font=dict(color=DARK_BLUE, family='Arial'),
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)

# Gráfico 2: Top Especialidades (CORRIGIDO: px.bar com orientation='h')
with col2:
    top_espec = df_filtrado['ESPEC_NOME'].value_counts().head(10).reset_index()
    top_espec.columns = ['Especialidade', 'Total']
    
    fig2 = px.bar(
        top_espec,
        y='Especialidade',
        x='Total',
        orientation='h',
        title='🏥 Top 10 Especialidades',
        color_discrete_sequence=[DARK_BLUE],
        text='Total'
    )
    fig2.update_traces(textposition='auto')
    fig2.update_layout(
        height=400,
        hovermode='y unified',
        paper_bgcolor=WHITE,
        font=dict(color=DARK_BLUE, family='Arial'),
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================================
# 📈 GRÁFICOS (LINHA 2)
# ============================================

col1, col2 = st.columns(2)

# Gráfico 3: Distribuição por Sexo
with col1:
    sexo_data = df_filtrado['SEXO_NOME'].value_counts()
    fig3 = px.pie(
        values=sexo_data.values,
        names=sexo_data.index,
        title='👥 Distribuição por Sexo',
        color_discrete_sequence=[TEAL, LIGHT_BLUE]
    )
    fig3.update_layout(
        height=400,
        paper_bgcolor=WHITE,
        font=dict(color=DARK_BLUE, family='Arial')
    )
    st.plotly_chart(fig3, use_container_width=True)

# Gráfico 4: Caráter da Internação
with col2:
    carater_data = df_filtrado['CAR_INT_NOME'].value_counts()
    fig4 = px.bar(
        x=carater_data.index,
        y=carater_data.values,
        title='🚨 Caráter da Internação',
        color_discrete_sequence=[DARK_BLUE],
        text=carater_data.values,
        labels={'x': 'Caráter', 'y': 'Total'}
    )
    fig4.update_traces(textposition='auto')
    fig4.update_layout(
        height=400,
        hovermode='x unified',
        paper_bgcolor=WHITE,
        font=dict(color=DARK_BLUE, family='Arial'),
        showlegend=False
    )
    st.plotly_chart(fig4, use_container_width=True)

# ============================================
# 📈 GRÁFICOS (LINHA 3)
# ============================================

col1, col2 = st.columns(2)

# Gráfico 5: Distribuição de Dias de Permanência
with col1:
    fig5 = go.Figure()
    fig5.add_trace(go.Histogram(
        x=df_filtrado['DIAS_PERM'],
        nbinsx=40,
        name='Dias',
        marker_color=TEAL,
        opacity=0.7
    ))
    fig5.add_vline(
        x=df_filtrado['DIAS_PERM'].mean(),
        line_dash="dash",
        line_color=DARK_BLUE,
        annotation_text=f"Média: {df_filtrado['DIAS_PERM'].mean():.1f}",
        annotation_position="top right"
    )
    fig5.update_layout(
        title='📅 Distribuição de Dias de Permanência',
        xaxis_title='Dias',
        yaxis_title='Frequência',
        height=400,
        paper_bgcolor=WHITE,
        font=dict(color=DARK_BLUE, family='Arial'),
        showlegend=False
    )
    st.plotly_chart(fig5, use_container_width=True)

# Gráfico 6: Custos por UF
with col2:
    custo_uf = df_filtrado.groupby('UF_ZI')['VAL_TOT'].mean().reset_index()
    custo_uf.columns = ['UF', 'Custo Médio']
    custo_uf = custo_uf.sort_values('Custo Médio', ascending=False)
    
    fig6 = px.bar(
        custo_uf,
        x='UF',
        y='Custo Médio',
        title='💰 Custo Médio por UF',
        color='Custo Médio',
        color_continuous_scale=[[0, LIGHT_BLUE], [1, DARK_BLUE]],
        text='Custo Médio'
    )
    fig6.update_traces(texttemplate='R$ %{y:,.0f}', textposition='auto')
    fig6.update_layout(
        height=400,
        hovermode='x unified',
        paper_bgcolor=WHITE,
        font=dict(color=DARK_BLUE, family='Arial'),
        coloraxis_colorbar=dict(title='R$')
    )
    st.plotly_chart(fig6, use_container_width=True)

# ============================================
# 📊 TABELA DE DADOS
# ============================================

st.markdown("---")
st.markdown("### 📋 Dados Detalhados")

colunas_disponiveis = ['N_AIH', 'DT_INTER', 'ESPEC_NOME', 'SEXO_NOME', 'IDADE', 'DIAS_PERM', 'VAL_TOT', 'CAR_INT_NOME', 'MORTE']
colunas_selecionadas = st.multiselect(
    "Colunas a exibir:",
    colunas_disponiveis,
    default=['N_AIH', 'DT_INTER', 'ESPEC_NOME', 'IDADE', 'DIAS_PERM', 'VAL_TOT']
)

st.dataframe(
    df_filtrado[colunas_selecionadas].head(20),
    use_container_width=True,
    height=500
)

# ============================================
# 💾 DOWNLOAD DE DADOS
# ============================================

st.markdown("---")
st.markdown("### 💾 Exportar Dados")

csv = df_filtrado.to_csv(index=False)
st.download_button(
    label="📥 Download CSV",
    data=csv,
    file_name=f"internacoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

# ============================================
# 📝 FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #505050; font-size: 0.9em; margin-top: 2em;'>
    <p>🚀 Dashboard desenvolvido com Streamlit</p>
    <p>📊 Desafio FIAP - Análise de Internações DATASUS</p>
    <p>💡 Dados de demonstração - Valores fictícios para teste</p>
    <p>Última atualização: {} UTC</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

# ============================================
# 🎯 INSTRUÇÕES DE DEPLOYMENT
# ============================================

if st.sidebar.checkbox("📖 Ver Instruções"):
    st.sidebar.markdown("""
    ### 🚀 Como fazer Deploy
    
    **Opção 1: Streamlit Cloud (Gratuito)**
    
    1. Salve este arquivo como `app.py`
    2. Crie `requirements.txt`:
    ```
    streamlit==1.25.0
    pandas==1.5.3
    plotly==5.15.0
    numpy==1.24.3
    ```
    3. Suba para GitHub
    4. Acesse share.streamlit.io
    5. Deploy!
    
    **Seu dashboard estará LIVE em 2 minutos!**
    
    [Mais informações](https://docs.streamlit.io/deploy/streamlit-community-cloud)
    """)
