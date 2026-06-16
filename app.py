# app.py - Dashboard DATASUS (3 GRÁFICOS PRINCIPAIS)
# 1. Exploração Inicial (Sazonalidade, Ranking, Municípios)
# 2. Indicadores de Capacidade (Permanência, Internações, Estrutura)
# 3. Padrões e Agrupamentos (Clustering + Perfis Críticos)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# ============================================
# 🎨 CONFIGURAÇÃO
# ============================================

st.set_page_config(
    page_title="Dashboard DATASUS",
    page_icon="🏥",
    layout="wide"
)

DARK_BLUE = '#1E2761'
TEAL = '#02B090'
LIGHT_BLUE = '#CADCFC'
WHITE = '#FFFFFF'

# ============================================
# 📊 CARREGAR DADOS
# ============================================

@st.cache_data
def load_data():
    """Gera dados de internações"""
    np.random.seed(42)
    n_records = 8000
    
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days
    random_dates = [start_date + timedelta(days=np.random.randint(0, date_range)) for _ in range(n_records)]
    
    ufs = ['SP', 'RJ', 'MG', 'BA', 'RS', 'PE', 'PR', 'SC', 'GO', 'DF']
    municipios_dict = {
        'SP': ['São Paulo', 'Campinas', 'Santos', 'Ribeirão Preto', 'Sorocaba'],
        'RJ': ['Rio de Janeiro', 'Niterói', 'Duque de Caxias', 'Nova Iguaçu'],
        'MG': ['Belo Horizonte', 'Uberlândia', 'Contagem', 'Juiz de Fora'],
        'BA': ['Salvador', 'Feira de Santana', 'Vitória da Conquista'],
        'RS': ['Porto Alegre', 'Caxias do Sul', 'Novo Hamburgo'],
        'PE': ['Recife', 'Jaboatão', 'Caruaru'],
        'PR': ['Curitiba', 'Londrina', 'Maringá'],
        'SC': ['Florianópolis', 'Blumenau', 'Joinville'],
        'GO': ['Goiânia', 'Anápolis'],
        'DF': ['Brasília']
    }
    
    especialidades = ['Clínica Médica', 'Cirurgia Geral', 'Ginecologia', 'Cardiologia', 
                     'Neurologia', 'Pediatria', 'Traumatologia', 'Oncologia']
    
    uf_choices = np.random.choice(ufs, n_records)
    municipios_list = [np.random.choice(municipios_dict[uf]) for uf in uf_choices]
    
    df = pd.DataFrame({
        'DT_INTER': random_dates,
        'UF_ZI': uf_choices,
        'MUNIC_RES': municipios_list,
        'ESPEC_NOME': np.random.choice(especialidades, n_records),
        'IDADE': np.random.gamma(shape=2, scale=20, size=n_records).astype(int),
        'DIAS_PERM': np.random.exponential(scale=5, size=n_records).astype(int) + 1,
        'CAR_INT': np.random.choice(['Eletiva', 'Urgência'], n_records, p=[0.4, 0.6]),
        'MORTE': np.random.binomial(1, 0.02, n_records),
        'VAL_SH': np.random.gamma(2, 500, n_records),
        'CNES': np.random.randint(1000000, 9999999, n_records),
        'LEITOS_DISP': np.random.randint(50, 500, n_records),
    })
    
    df['MES'] = df['DT_INTER'].dt.month
    df['ANO'] = df['DT_INTER'].dt.year
    
    return df

df = load_data()

# ============================================
# 📱 TÍTULO
# ============================================

st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #1E2761;
    font-size: 2.5em;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #505050;
    font-size: 1.1em;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🏥 Dashboard DATASUS</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Análise de Internações Hospitalares</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# 🔍 FILTROS
# ============================================

st.sidebar.title("🔍 Filtros")

ufs_selecionadas = st.sidebar.multiselect(
    "Selecione UF:",
    sorted(df['UF_ZI'].unique()),
    default=sorted(df['UF_ZI'].unique())
)

df_filtrado = df[df['UF_ZI'].isin(ufs_selecionadas)]

st.sidebar.info(f"📊 Registros: {len(df_filtrado):,}")

# ============================================
# 📊 KPIs
# ============================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Internações", f"{len(df_filtrado):,}")

with col2:
    taxa_mort = (df_filtrado['MORTE'].sum() / len(df_filtrado) * 100) if len(df_filtrado) > 0 else 0
    st.metric("Taxa Mortalidade", f"{taxa_mort:.2f}%")

with col3:
    dias_med = df_filtrado['DIAS_PERM'].mean()
    st.metric("Permanência Média", f"{dias_med:.1f} dias")

with col4:
    leitos_med = df_filtrado['LEITOS_DISP'].mean()
    st.metric("Leitos Disponíveis", f"{leitos_med:.0f}")

st.markdown("---")

# ============================================
# GRÁFICO 1: EXPLORAÇÃO INICIAL
# ============================================

st.markdown("## 📊 1. EXPLORAÇÃO INICIAL")
st.markdown("*Sazonalidade | Ranking | Municípios*")

# Criar figura com subplots
from plotly.subplots import make_subplots

fig1 = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "📈 Sazonalidade: Internações por Mês",
        "🥇 Top 5 Especialidades",
        "🏘️ Top 5 Municípios",
        "🚨 Taxa Mortalidade por Especialidade"
    ),
    specs=[
        [{"type": "scatter"}, {"type": "bar"}],
        [{"type": "bar"}, {"type": "bar"}]
    ]
)

# Subplot 1: Sazonalidade
meses_nome = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
sazonalidade = df_filtrado.groupby('MES').size().reset_index(name='Total')
sazonalidade['Mês'] = sazonalidade['MES'].apply(lambda x: meses_nome[x-1] if x <= 12 else 'N/A')

fig1.add_trace(
    go.Scatter(
        x=sazonalidade['Mês'],
        y=sazonalidade['Total'],
        mode='lines+markers',
        name='Internações',
        line=dict(color=TEAL, width=3),
        marker=dict(size=8)
    ),
    row=1, col=1
)

# Subplot 2: Top Especialidades
top_espec = df_filtrado['ESPEC_NOME'].value_counts().head(5).reset_index()
top_espec.columns = ['Especialidade', 'Total']

fig1.add_trace(
    go.Bar(
        x=top_espec['Especialidade'],
        y=top_espec['Total'],
        name='Internações',
        marker=dict(color=TEAL),
        text=top_espec['Total'],
        textposition='auto'
    ),
    row=1, col=2
)

# Subplot 3: Top Municípios
top_mun = df_filtrado['MUNIC_RES'].value_counts().head(5).reset_index()
top_mun.columns = ['Município', 'Total']

fig1.add_trace(
    go.Bar(
        x=top_mun['Município'],
        y=top_mun['Total'],
        name='Internações',
        marker=dict(color=LIGHT_BLUE),
        text=top_mun['Total'],
        textposition='auto'
    ),
    row=2, col=1
)

# Subplot 4: Taxa Mortalidade por Especialidade
mort_espec = df_filtrado.groupby('ESPEC_NOME').agg({
    'MORTE': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
}).reset_index()
mort_espec.columns = ['Especialidade', 'Taxa %']
mort_espec = mort_espec.sort_values('Taxa %', ascending=False).head(5)

fig1.add_trace(
    go.Bar(
        x=mort_espec['Especialidade'],
        y=mort_espec['Taxa %'],
        name='Taxa %',
        marker=dict(color='#F96167'),
        text=mort_espec['Taxa %'].round(2),
        texttemplate='%{text:.2f}%',
        textposition='auto'
    ),
    row=2, col=2
)

# Atualizar layout
fig1.update_xaxes(title_text="Mês", row=1, col=1)
fig1.update_yaxes(title_text="Total", row=1, col=1)
fig1.update_xaxes(tickangle=-45, row=1, col=2)
fig1.update_xaxes(tickangle=-45, row=2, col=1)
fig1.update_xaxes(tickangle=-45, row=2, col=2)

fig1.update_layout(
    height=700,
    showlegend=False,
    paper_bgcolor=WHITE,
    font=dict(color=DARK_BLUE, family='Arial'),
    title_text="<b>📊 Exploração Inicial: Sazonalidade, Rankings e Comparações</b>",
    title_x=0.5
)

st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ============================================
# GRÁFICO 2: INDICADORES DE CAPACIDADE
# ============================================

st.markdown("## 📈 2. INDICADORES DE CAPACIDADE")
st.markdown("*Permanência Média | Internações | Estrutura Disponível*")

fig2 = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        "⏱️ Permanência Média por UF",
        "📊 Volume de Internações por UF",
        "🏗️ Leitos vs Demanda"
    ),
    specs=[
        [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}]
    ]
)

# Dados por UF
capacidade_uf = df_filtrado.groupby('UF_ZI').agg({
    'DIAS_PERM': 'mean',
    'MUNIC_RES': 'count',
    'LEITOS_DISP': 'mean'
}).reset_index()
capacidade_uf.columns = ['UF', 'Perm_Media', 'Internacoes', 'Leitos']

# Subplot 1: Permanência Média
fig2.add_trace(
    go.Bar(
        x=capacidade_uf['UF'],
        y=capacidade_uf['Perm_Media'],
        name='Dias',
        marker=dict(color=TEAL),
        text=capacidade_uf['Perm_Media'].round(1),
        textposition='auto'
    ),
    row=1, col=1
)

# Subplot 2: Volume de Internações
fig2.add_trace(
    go.Bar(
        x=capacidade_uf['UF'],
        y=capacidade_uf['Internacoes'],
        name='Total',
        marker=dict(color=LIGHT_BLUE),
        text=capacidade_uf['Internacoes'],
        textposition='auto'
    ),
    row=1, col=2
)

# Subplot 3: Leitos vs Demanda (Scatter)
fig2.add_trace(
    go.Scatter(
        x=capacidade_uf['Leitos'],
        y=capacidade_uf['Internacoes'],
        mode='markers+text',
        text=capacidade_uf['UF'],
        textposition='top center',
        marker=dict(
            size=12,
            color=capacidade_uf['Perm_Media'],
            colorscale=[[0, TEAL], [1, '#F96167']],
            showscale=True,
            colorbar=dict(title="Permanência")
        ),
        name='UF'
    ),
    row=1, col=3
)

fig2.update_xaxes(title_text="UF", row=1, col=1)
fig2.update_xaxes(title_text="UF", row=1, col=2)
fig2.update_xaxes(title_text="Leitos Disponíveis", row=1, col=3)
fig2.update_yaxes(title_text="Dias", row=1, col=1)
fig2.update_yaxes(title_text="Internações", row=1, col=2)
fig2.update_yaxes(title_text="Internações", row=1, col=3)

fig2.update_layout(
    height=500,
    showlegend=False,
    paper_bgcolor=WHITE,
    font=dict(color=DARK_BLUE, family='Arial'),
    title_text="<b>📈 Indicadores de Capacidade: Permanência, Volume e Estrutura</b>",
    title_x=0.5
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ============================================
# GRÁFICO 3: CLUSTERING + PERFIS CRÍTICOS
# ============================================

st.markdown("## 🎯 3. PADRÕES E AGRUPAMENTOS")
st.markdown("*Clusterização de Hospitais | Perfis Críticos*")

# Preparar dados para clustering
hospital_data = df_filtrado.groupby('CNES').agg({
    'MUNIC_RES': 'first',
    'DIAS_PERM': ['mean', 'std'],
    'MORTE': ['sum', 'mean'],
    'LEITOS_DISP': 'mean',
    'IDADE': 'mean',
    'UF_ZI': 'first'
}).reset_index()

hospital_data.columns = ['CNES', 'Municipio', 'Perm_Media', 'Perm_Std', 'Mortes', 'Taxa_Mort', 'Leitos', 'Idade_Media', 'UF']
hospital_data['Perm_Std'] = hospital_data['Perm_Std'].fillna(0)

# Contar internações por hospital
internacoes_hosp = df_filtrado.groupby('CNES').size().reset_index(name='Internacoes')
hospital_data = hospital_data.merge(internacoes_hosp, on='CNES', how='left')

# Clustering
features = hospital_data[['Perm_Media', 'Taxa_Mort', 'Leitos', 'Internacoes']].fillna(0)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
hospital_data['Cluster'] = kmeans.fit_predict(features_scaled)

# Nomear clusters
cluster_names = {
    0: 'Padrão Normal',
    1: 'Padrão Crítico',
    2: 'Padrão Especial'
}
cluster_colors = {
    0: TEAL,
    1: '#F96167',
    2: LIGHT_BLUE
}

hospital_data['Cluster_Nome'] = hospital_data['Cluster'].map(cluster_names)
hospital_data['Cor'] = hospital_data['Cluster'].map(cluster_colors)

# Criar figura com subplots
fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "🏥 Clustering: Permanência vs Mortalidade",
        "📊 Distribuição dos Clusters"
    ),
    specs=[
        [{"type": "scatter"}, {"type": "pie"}]
    ]
)

# Scatter: Clustering
for cluster in hospital_data['Cluster'].unique():
    cluster_data = hospital_data[hospital_data['Cluster'] == cluster]
    fig3.add_trace(
        go.Scatter(
            x=cluster_data['Perm_Media'],
            y=cluster_data['Taxa_Mort'],
            mode='markers',
            name=cluster_names[cluster],
            marker=dict(
                size=cluster_data['Internacoes'] / 2,
                color=cluster_colors[cluster],
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=cluster_data['Municipio'],
            hovertemplate='<b>%{text}</b><br>Permanência: %{x:.1f} dias<br>Taxa Mortalidade: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )

# Pizza: Distribuição
cluster_dist = hospital_data['Cluster_Nome'].value_counts().reset_index()
cluster_dist.columns = ['Cluster', 'Quantidade']

fig3.add_trace(
    go.Pie(
        labels=cluster_dist['Cluster'],
        values=cluster_dist['Quantidade'],
        marker=dict(colors=[TEAL, '#F96167', LIGHT_BLUE]),
        textposition='inside',
        textinfo='label+percent'
    ),
    row=1, col=2
)

fig3.update_xaxes(title_text="Permanência Média (dias)", row=1, col=1)
fig3.update_yaxes(title_text="Taxa de Mortalidade (%)", row=1, col=1)

fig3.update_layout(
    height=550,
    paper_bgcolor=WHITE,
    font=dict(color=DARK_BLUE, family='Arial'),
    title_text="<b>🎯 Padrões e Agrupamentos: Machine Learning</b>",
    title_x=0.5,
    showlegend=True
)

st.plotly_chart(fig3, use_container_width=True)

# Perfis Críticos
st.markdown("### 🚨 Perfis Críticos Identificados")

col1, col2, col3 = st.columns(3)

with col1:
    hosp_critico = hospital_data.loc[hospital_data['Taxa_Mort'].idxmax()]
    st.metric(
        "Maior Taxa Mortalidade",
        f"{hosp_critico['Municipio']}",
        f"{hosp_critico['Taxa_Mort']:.2f}%"
    )

with col2:
    hosp_perm = hospital_data.loc[hospital_data['Perm_Media'].idxmax()]
    st.metric(
        "Maior Permanência",
        f"{hosp_perm['Municipio']}",
        f"{hosp_perm['Perm_Media']:.1f} dias"
    )

with col3:
    hosp_vol = hospital_data.loc[hospital_data['Internacoes'].idxmax()]
    st.metric(
        "Maior Volume",
        f"{hosp_vol['Municipio']}",
        f"{int(hosp_vol['Internacoes'])} internações"
    )

st.markdown("---")

# ============================================
# FOOTER
# ============================================

st.markdown("""
<div style='text-align: center; color: #505050; margin-top: 2em;'>
    <p>🚀 Dashboard DATASUS - Análise de Internações</p>
    <p>3 Gráficos Principais: Exploração | Capacidade | Agrupamentos</p>
</div>
""", unsafe_allow_html=True)
