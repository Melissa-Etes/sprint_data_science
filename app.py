# app.py - Dashboard Internações DATASUS (NOVO)
# Gráficos específicos: Sazonalidade, Ranking, Municípios, Capacidade, Clustering

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
    page_title="Dashboard DATASUS - Análise Avançada",
    page_icon="🏥",
    layout="wide"
)

DARK_BLUE = '#1E2761'
TEAL = '#02B090'
LIGHT_BLUE = '#CADCFC'
WHITE = '#FFFFFF'
GRAY = '#505050'

# ============================================
# 📊 CARREGAR DADOS
# ============================================

@st.cache_data
def load_data():
    """Gera dados realistas de internações"""
    np.random.seed(42)
    n_records = 8000
    
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days
    random_dates = [start_date + timedelta(days=np.random.randint(0, date_range)) for _ in range(n_records)]
    
    ufs = ['SP', 'RJ', 'MG', 'BA', 'RS', 'PE', 'PR', 'SC', 'GO', 'DF']
    municipios = {
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
    
    uf_choices = np.random.choice(ufs, n_records)
    
    municipios_list = []
    for uf in uf_choices:
        municipios_list.append(np.random.choice(municipios[uf]))
    
    df = pd.DataFrame({
        'DT_INTER': random_dates,
        'UF_ZI': uf_choices,
        'MUNIC_RES': municipios_list,
        'ESPEC': np.random.choice(list(especialidades.keys()), n_records),
        'ESPEC_NOME': [especialidades.get(str(x), 'Outra') for x in np.random.choice(list(especialidades.keys()), n_records)],
        'SEXO': np.random.choice(['M', 'F'], n_records),
        'IDADE': np.random.gamma(shape=2, scale=20, size=n_records).astype(int),
        'DIAS_PERM': np.random.exponential(scale=5, size=n_records).astype(int) + 1,
        'CAR_INT': np.random.choice(['Eletiva', 'Urgência'], n_records, p=[0.4, 0.6]),
        'MORTE': np.random.binomial(1, 0.02, n_records),
        'VAL_SH': np.random.gamma(2, 500, n_records),
        'VAL_SP': np.random.gamma(2, 300, n_records),
        'CNES': np.random.randint(1000000, 9999999, n_records),
        'LEITOS_DISP': np.random.randint(50, 500, n_records),
    })
    
    df['VAL_TOT'] = df['VAL_SH'] + df['VAL_SP']
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
    margin-bottom: 0.3em;
}
.subtitle {
    text-align: center;
    color: #505050;
    font-size: 1.1em;
    margin-bottom: 1em;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🏥 Dashboard DATASUS - Análise Avançada</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Internações Hospitalares: Exploração, Capacidade e Agrupamentos</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# 🔍 FILTROS
# ============================================

st.sidebar.title("🔍 Filtros")

ufs_selecionadas = st.sidebar.multiselect(
    "Selecione UF:",
    sorted(df['UF_ZI'].unique()),
    default=sorted(df['UF_ZI'].unique())[:5]
)

df_filtrado = df[df['UF_ZI'].isin(ufs_selecionadas)]

st.sidebar.info(f"📊 Registros: {len(df_filtrado):,} de {len(df):,}")

# ============================================
# 📊 SEÇÃO 1: EXPLORAÇÃO INICIAL
# ============================================

st.markdown("## 📊 1. EXPLORAÇÃO INICIAL")
st.markdown("---")

col1, col2, col3 = st.columns(3)

# KPIs
with col1:
    st.metric("Total de Internações", f"{len(df_filtrado):,}")

with col2:
    taxa_mort = (df_filtrado['MORTE'].sum() / len(df_filtrado) * 100) if len(df_filtrado) > 0 else 0
    st.metric("Taxa de Mortalidade", f"{taxa_mort:.2f}%")

with col3:
    dias_medio = df_filtrado['DIAS_PERM'].mean()
    st.metric("Permanência Média (dias)", f"{dias_medio:.1f}")

st.markdown("")

# ### 1.1 Sazonalidade
col1, col2 = st.columns(2)

with col1:
    # Sazonalidade por mês
    sazonalidade = df_filtrado.groupby('MES').size().reset_index(name='Total')
    meses_nome = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    sazonalidade['Mês'] = sazonalidade['MES'].apply(lambda x: meses_nome[x-1] if x <= 12 else 'N/A')
    
    fig1 = px.line(
        sazonalidade,
        x='Mês',
        y='Total',
        title='📈 Sazonalidade: Internações por Mês',
        markers=True,
        color_discrete_sequence=[TEAL]
    )
    fig1.update_layout(
        height=400,
        hovermode='x unified',
        paper_bgcolor=WHITE,
        font=dict(color=DARK_BLUE)
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Sazonalidade por tipo de internação
    carater_mes = df_filtrado.groupby(['MES', 'CAR_INT']).size().reset_index(name='Total')
    carater_mes['Mês'] = carater_mes['MES'].apply(lambda x: meses_nome[x-1] if x <= 12 else 'N/A')
    
    fig2 = px.bar(
        carater_mes,
        x='Mês',
        y='Total',
        color='CAR_INT',
        title='🏥 Sazonalidade: Eletiva vs Urgência',
        color_discrete_sequence=[TEAL, '#F96167'],
        barmode='stack'
    )
    fig2.update_layout(
        height=400,
        paper_bgcolor=WHITE,
        font=dict(color=DARK_BLUE)
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("")

# ### 1.2 Ranking de Especialidades
col1, col2 = st.columns(2)

with col1:
    # Top especialidades
    top_espec = df_filtrado['ESPEC_NOME'].value_counts().head(8).reset_index()
    top_espec.columns = ['Especialidade', 'Total']
    
    fig3 = px.bar(
        top_espec,
        y='Especialidade',
        x='Total',
        orientation='h',
        title='🥇 Ranking: Top 8 Especialidades',
        color='Total',
        color_continuous_scale=[[0, LIGHT_BLUE], [1, DARK_BLUE]],
        text='Total'
    )
    fig3.update_traces(textposition='auto')
    fig3.update_layout(
        height=400,
        paper_bgcolor=WHITE,
        font=dict(color=DARK_BLUE),
        showlegend=False
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    # Ranking por taxa de mortalidade por especialidade
    mort_espec = df_filtrado.groupby('ESPEC_NOME').agg({
        'MORTE': lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0
    }).reset_index()
    mort_espec.columns = ['Especialidade', 'Taxa Mortalidade %']
    mort_espec = mort_espec.sort_values('Taxa Mortalidade %', ascending=False).head(8)
    
    fig4 = px.bar(
        mort_espec,
        x='Especialidade',
        y='Taxa Mortalidade %',
        title='⚠️ Ranking: Taxa de Mortalidade por Especialidade',
        color='Taxa Mortalidade %',
        color_continuous_scale=[[0, TEAL], [1, '#F96167']],
        text='Taxa Mortalidade %'
    )
    fig4.update_traces(texttemplate='%{y:.2f}%', textposition='auto')
    fig4.update_layout(
        height=400,
        paper_bgcolor=WHITE,
        font=dict(color=DARK_BLUE),
        xaxis_tickangle=-45,
        showlegend=False
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("")

# ### 1.3 Comparação entre Municípios
municipios_internacoes = df_filtrado.groupby('MUNIC_RES').size().reset_index(name='Total').sort_values('Total', ascending=False).head(10)

fig5 = px.bar(
    municipios_internacoes,
    y='MUNIC_RES',
    x='Total',
    orientation='h',
    title='🏘️ Comparação: Top 10 Municípios com Mais Internações',
    color='Total',
    color_continuous_scale=[[0, LIGHT_BLUE], [1, TEAL]],
    text='Total'
)
fig5.update_traces(textposition='auto')
fig5.update_layout(
    height=450,
    paper_bgcolor=WHITE,
    font=dict(color=DARK_BLUE),
    showlegend=False
)
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# ============================================
# 📈 SEÇÃO 2: INDICADORES DE CAPACIDADE
# ============================================

st.markdown("## 📈 2. INDICADORES DE CAPACIDADE")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Permanência Média", f"{df_filtrado['DIAS_PERM'].mean():.1f} dias")

with col2:
    st.metric("Total de Internações", f"{len(df_filtrado):,}")

with col3:
    st.metric("Leitos Médios Disponíveis", f"{df_filtrado['LEITOS_DISP'].mean():.0f}")

st.markdown("")

# ### 2.1 Permanência Média por Especialidade
permanencia_espec = df_filtrado.groupby('ESPEC_NOME')['DIAS_PERM'].agg(['mean', 'count']).reset_index()
permanencia_espec.columns = ['Especialidade', 'Permanência Média', 'Internações']
permanencia_espec = permanencia_espec.sort_values('Permanência Média', ascending=False).head(10)

fig6 = px.bar(
    permanencia_espec,
    x='Especialidade',
    y='Permanência Média',
    color='Permanência Média',
    color_continuous_scale=[[0, TEAL], [1, '#F96167']],
    title='⏱️ Permanência Média por Especialidade (Top 10)',
    text='Permanência Média'
)
fig6.update_traces(texttemplate='%{y:.1f} dias', textposition='auto')
fig6.update_layout(
    height=400,
    paper_bgcolor=WHITE,
    font=dict(color=DARK_BLUE),
    xaxis_tickangle=-45,
    showlegend=False
)
st.plotly_chart(fig6, use_container_width=True)

st.markdown("")

# ### 2.2 Volume de Internações vs Estrutura Disponível
vol_estrutura = df_filtrado.groupby('UF_ZI').agg({
    'MUNIC_RES': 'count',
    'LEITOS_DISP': 'mean'
}).reset_index()
vol_estrutura.columns = ['UF', 'Internações', 'Leitos Médios']

fig7 = px.scatter(
    vol_estrutura,
    x='Leitos Médios',
    y='Internações',
    color='Leitos Médios',
    size='Internações',
    text='UF',
    title='🏗️ Estrutura vs Demanda: Leitos Disponíveis vs Internações',
    color_continuous_scale=[[0, LIGHT_BLUE], [1, DARK_BLUE]],
    size_max=50
)
fig7.update_traces(textposition='top center')
fig7.update_layout(
    height=450,
    paper_bgcolor=WHITE,
    font=dict(color=DARK_BLUE),
    hovermode='closest'
)
st.plotly_chart(fig7, use_container_width=True)

st.markdown("")

# ### 2.3 Capacidade por UF
capacidade_uf = df_filtrado.groupby('UF_ZI').agg({
    'MUNIC_RES': 'count',
    'DIAS_PERM': 'mean'
}).reset_index()
capacidade_uf.columns = ['UF', 'Internações', 'Permanência Média']
capacidade_uf = capacidade_uf.sort_values('Internações', ascending=False)

fig8 = px.bar(
    capacidade_uf,
    x='UF',
    y='Internações',
    color='Permanência Média',
    color_continuous_scale=[[0, TEAL], [1, '#F96167']],
    title='🏥 Capacidade: Internações por UF (colorido por Permanência Média)',
    text='Internações'
)
fig8.update_traces(textposition='auto')
fig8.update_layout(
    height=400,
    paper_bgcolor=WHITE,
    font=dict(color=DARK_BLUE),
    showlegend=True
)
st.plotly_chart(fig8, use_container_width=True)

st.markdown("---")

# ============================================
# 🎯 SEÇÃO 3: PADRÕES E AGRUPAMENTOS
# ============================================

st.markdown("## 🎯 3. PADRÕES E AGRUPAMENTOS")
st.markdown("---")

# ### 3.1 Clusterização de Hospitais
# Preparar dados para clustering por CNES
hospital_features = df_filtrado.groupby('CNES').agg({
    'MUNIC_RES': 'first',
    'DIAS_PERM': ['mean', 'std'],
    'MORTE': ['sum', 'mean'],
    'LEITOS_DISP': 'mean',
    'IDADE': 'mean',
    'ESPEC_NOME': 'count'
}).reset_index()

hospital_features.columns = ['CNES', 'Municipio', 'Perm_Media', 'Perm_Std', 'Mortes_Total', 'Taxa_Mortalidade', 'Leitos_Media', 'Idade_Media', 'Internacoes']

hospital_features['Perm_Std'] = hospital_features['Perm_Std'].fillna(0)

# Features para clustering
features_for_clustering = hospital_features[['Perm_Media', 'Taxa_Mortalidade', 'Leitos_Media', 'Internacoes']].fillna(0)

# Normalizar
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features_for_clustering)

# Clustering (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
hospital_features['Cluster'] = kmeans.fit_predict(features_normalized)

cluster_names = {
    0: 'Cluster A: Padrão Normal',
    1: 'Cluster B: Padrão Crítico',
    2: 'Cluster C: Padrão Especial'
}

hospital_features['Cluster_Nome'] = hospital_features['Cluster'].map(cluster_names)

# Gráfico de clustering
fig9 = px.scatter(
    hospital_features,
    x='Perm_Media',
    y='Taxa_Mortalidade',
    color='Cluster_Nome',
    size='Internacoes',
    text='Municipio',
    title='🏥 Clusterização de Hospitais por Padrão de Internação',
    color_discrete_map={
        'Cluster A: Padrão Normal': TEAL,
        'Cluster B: Padrão Crítico': '#F96167',
        'Cluster C: Padrão Especial': LIGHT_BLUE
    },
    size_max=40,
    hover_data=['Leitos_Media', 'Internacoes']
)
fig9.update_traces(textposition='top center')
fig9.update_layout(
    height=500,
    paper_bgcolor=WHITE,
    font=dict(color=DARK_BLUE),
    hovermode='closest'
)
st.plotly_chart(fig9, use_container_width=True)

st.markdown("")

# ### 3.2 Perfis Críticos
st.markdown("#### 🚨 Perfis Críticos Identificados")

# Hospital com maior taxa de mortalidade
hospital_critico = hospital_features.loc[hospital_features['Taxa_Mortalidade'].idxmax()]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Hospital Mais Crítico",
        f"{hospital_critico['Municipio']}",
        f"Taxa: {hospital_critico['Taxa_Mortalidade']:.2f}%"
    )

with col2:
    hospital_maior_perm = hospital_features.loc[hospital_features['Perm_Media'].idxmax()]
    st.metric(
        "Hospital Maior Permanência",
        f"{hospital_maior_perm['Municipio']}",
        f"Média: {hospital_maior_perm['Perm_Media']:.1f} dias"
    )

with col3:
    hospital_maior_vol = hospital_features.loc[hospital_features['Internacoes'].idxmax()]
    st.metric(
        "Hospital Maior Volume",
        f"{hospital_maior_vol['Municipio']}",
        f"Total: {int(hospital_maior_vol['Internacoes'])} internações"
    )

st.markdown("")

# ### 3.3 Distribuição dos Clusters
cluster_dist = hospital_features['Cluster_Nome'].value_counts().reset_index()
cluster_dist.columns = ['Cluster', 'Quantidade']

fig10 = px.pie(
    cluster_dist,
    names='Cluster',
    values='Quantidade',
    title='📊 Distribuição dos Hospitais por Cluster',
    color_discrete_map={
        'Cluster A: Padrão Normal': TEAL,
        'Cluster B: Padrão Crítico': '#F96167',
        'Cluster C: Padrão Especial': LIGHT_BLUE
    }
)
fig10.update_layout(
    height=400,
    paper_bgcolor=WHITE,
    font=dict(color=DARK_BLUE)
)
st.plotly_chart(fig10, use_container_width=True)

st.markdown("---")

# ============================================
# 📋 DADOS DETALHADOS
# ============================================

st.markdown("### 📋 Hospitais por Cluster (Padrões Identificados)")

hospital_display = hospital_features[[
    'Municipio', 'Cluster_Nome', 'Internacoes', 'Perm_Media', 
    'Taxa_Mortalidade', 'Leitos_Media'
]].copy()
hospital_display.columns = ['Município', 'Padrão', 'Internações', 'Permanência Média', 'Taxa Mortalidade %', 'Leitos Médios']

st.dataframe(
    hospital_display.sort_values('Internações', ascending=False),
    use_container_width=True,
    height=500
)

st.markdown("---")

# ============================================
# FOOTER
# ============================================

st.markdown("""
<div style='text-align: center; color: #505050; font-size: 0.9em; margin-top: 2em;'>
    <p>🚀 Dashboard Avançado de Internações - DATASUS</p>
    <p>Análise: Sazonalidade | Capacidade | Agrupamentos</p>
</div>
""", unsafe_allow_html=True)
